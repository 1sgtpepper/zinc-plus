#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::projections::{ColumnMajorTrace, RowMajorTrace};
use crypto_primitives::PrimeField;
use num_traits::Zero;
use std::collections::HashMap;
use zinc_poly::{
    EvaluationError,
    mle::{
        DenseMultilinearExtension, MultilinearExtensionWithConfig, dense::CollectDenseMleWithZero,
    },
    univariate::dynamic::over_field::DynamicPolynomialF,
};
use zinc_uair::{
    ColumnLayout, ConstraintBuilder, ConstraintRing, TraceRow, Uair,
    degree_counter::{count_constraint_degrees, count_max_degree},
    ideal::ImpossibleIdeal,
};
use zinc_utils::{
    cfg_into_iter, cfg_iter, from_ref::FromRef, inner_transparent_field::InnerTransparentField,
};

/// Given a UAIR `U` and a trace `trace` this function
/// obtains the combined polynomials' MLE coefficients.
/// Since each coefficient is also a univariate polynomial
/// we split the resulting MLE into coefficient MLEs.
///
/// `trace_matrix` is row-indexed: `trace_matrix[row][col]`.
#[allow(clippy::arithmetic_side_effects)]
pub fn compute_combined_polynomials<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    skip_constraints: &[bool],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    F: PrimeField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    let num_rows = trace_matrix.len();

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];

                let down: Vec<DynamicPolynomialF<F>> = uair_sig
                    .shifts()
                    .iter()
                    .map(|spec| {
                        if row_idx + spec.shift_amount() < num_rows {
                            trace_matrix[row_idx + spec.shift_amount()][spec.source_col()].clone()
                        } else {
                            DynamicPolynomialF::zero() // zero padding
                        }
                    })
                    .collect();

                combine_rows_and_get_max_degree::<F, U>(
                    up,
                    &down,
                    num_constraints,
                    projected_scalars,
                    down_layout,
                )
            })
            .collect();

    let max_degree = *max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree)
        .max()
        .expect("We assume the number of constraints is not zero so this iterator is not empty");

    // For the sake of padding we duplicate
    // the last combined value
    // to have N-sized mle at the end
    // not N-1.
    // This is essentially c^up and c^down
    // thing from the whirlaway.
    // TODO(Ilia): reimplement it using Albert's idea
    //             with selector polynomials.
    max_degrees_and_combined_poly_rows.push((0, vec![DynamicPolynomialF::zero(); num_constraints]));

    prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        field_zero.inner(),
        skip_constraints,
    )
}

/// Apply combination polynomial to each row
/// and compute the maximum degree of resulting polynomials
/// to pad the resulting vector of MLEs accordingly.
#[allow(clippy::arithmetic_side_effects)]
fn combine_rows_and_get_max_degree<F, U>(
    up: &[DynamicPolynomialF<F>],
    down: &[DynamicPolynomialF<F>],
    num_constraints: usize,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    down_layout: &ColumnLayout,
) -> (usize, Vec<DynamicPolynomialF<F>>)
where
    F: PrimeField,
    U: Uair,
{
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(up, U::signature().total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout(down, down_layout),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;

    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    let max_degree = combined_evaluations
        .iter()
        .map(|eval| eval.degree().unwrap_or(0))
        .max()
        .expect("We assume the number of constraints is not zero so this iterator is not empty");

    (max_degree, combined_evaluations)
}

/// Turn the resulting slice of vectors of dynamic polynomials
/// into a vector of vectors of coefficient MLEs.
fn prepare_coefficient_mles<F: PrimeField>(
    num_constraints: usize,
    max_degree: usize,
    max_degrees_and_combined_poly_rows: &[(usize, Vec<DynamicPolynomialF<F>>)],
    zero_as_field_inner: &F::Inner,
    skip_constraints: &[bool],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>> {
    cfg_into_iter!(0..num_constraints)
        .map(|constraint| {
            // Skip building coefficient MLEs for zero-ideal constraints.
            // For an honest prover these MLEs are zero; the combined
            // polynomial resolver handles the zero entries downstream.
            if skip_constraints[constraint] {
                return vec![];
            }
            (0..=max_degree)
                .map(|coeff| {
                    max_degrees_and_combined_poly_rows
                        .iter()
                        .map(|(_, row)| {
                            if coeff >= row[constraint].coeffs.len() {
                                zero_as_field_inner.clone()
                            } else {
                                row[constraint].coeffs[coeff].inner().clone()
                            }
                        })
                        .collect_dense_mle_with_zero(zero_as_field_inner)
                })
                .collect()
        })
        .collect()
}

/// For linear UAIRs, evaluate combined polynomials directly
/// by first evaluating trace column MLEs at the evaluation point,
/// then applying UAIR constraints to the evaluated values.
///
/// This avoids building the full combined polynomial MLEs row by row
/// and is more efficient for linear constraints because the evaluation
/// of a linear combination of MLEs equals the linear combination of
/// individual MLE evaluations.
///
/// `trace_matrix` is column-indexed: `trace_matrix[col]` is an MLE.
///
/// Does `(num_columns + num_shifted_columns) * max_num_coeffs` evaluations of
/// MLEs.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    // Sanity check: this approach only works for linear constraints
    if count_max_degree::<U>() > 1 {
        return Err(EvaluationError::UnsupportedConstraintDegrees {
            degrees: count_constraint_degrees::<U>(),
        });
    }
    evaluate_combined_polynomials_unchecked::<F, U>(
        trace_matrix,
        projected_scalars,
        num_constraints,
        evaluation_point,
        field_cfg,
    )
}

/// Like [`evaluate_combined_polynomials`] but skips the global linearity
/// check. The caller is responsible for guaranteeing that values
/// returned for any non-linear, non-zero-ideal slot are discarded —
/// they would otherwise corrupt the transcript and break soundness.
///
/// Used by the dual-prime tag-aware ideal-check dispatch to compute
/// MLE-first values for the linear (resp. on-tag) subset of a
/// mixed-degree UAIR; off-tag slots are zeroed by the caller before
/// transcript absorption.
#[allow(clippy::arithmetic_side_effects)]
pub fn evaluate_combined_polynomials_unchecked<F, U>(
    trace_matrix: &ColumnMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    evaluation_point: &[F],
    field_cfg: &F::Config,
) -> Result<Vec<DynamicPolynomialF<F>>, EvaluationError>
where
    F: InnerTransparentField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let zero_inner = field_zero.inner().clone();
    let num_rows = trace_matrix.first().map(|c| c.len()).unwrap_or(0);
    let num_vars = evaluation_point.len();

    // Maximum number of coefficients across all trace entries
    let max_num_coeffs = trace_matrix
        .iter()
        .flat_map(|col| col.evaluations.iter())
        .map(|p| p.coeffs.len())
        .max()
        .unwrap_or(0);

    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();

    // Helper: evaluate one column's coefficient-d MLE at `evaluation_point`,
    // reading row `i + shift` (zero-padded beyond trace length).
    let eval_coeff_mle = |col: &DenseMultilinearExtension<DynamicPolynomialF<F>>,
                          d: usize,
                          shift: usize|
     -> Result<F, EvaluationError> {
        let coeff_evals: Vec<F::Inner> = (0..num_rows)
            .map(|i| {
                // Two conditions needed:
                // 1. i < num_rows - 1: zero out the last row for all columns (both up and down)
                //    to match the combined poly builder's explicit zero-padding at row N-1.
                // 2. i + shift < num_rows: prevent OOB access for shifts > 0.
                if i < num_rows - 1 && i + shift < num_rows {
                    col.evaluations[i + shift]
                        .coeffs
                        .get(d)
                        .map(|c| c.inner().clone())
                        .unwrap_or_else(|| zero_inner.clone())
                } else {
                    zero_inner.clone()
                }
            })
            .collect();
        let coeff_mle = DenseMultilinearExtension {
            evaluations: coeff_evals,
            num_vars,
        };
        coeff_mle.evaluate_with_config(evaluation_point, field_cfg)
    };

    // Evaluate up (all columns, shift=0).
    let up_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(trace_matrix)
        .map(|col| {
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, 0))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    // Evaluate down (only shifted columns, per-spec shift amount).
    let sorted_shifts = uair_sig.shifts();
    let down_evals: Vec<DynamicPolynomialF<F>> = cfg_iter!(sorted_shifts)
        .map(|spec| {
            let col = &trace_matrix[spec.source_col()];
            let coeffs: Vec<F> = (0..max_num_coeffs)
                .map(|d| eval_coeff_mle(col, d, spec.shift_amount()))
                .collect::<Result<_, _>>()?;
            Ok(DynamicPolynomialF::new_trimmed(coeffs))
        })
        .collect::<Result<Vec<_>, EvaluationError>>()?;

    // Apply UAIR constraints to the evaluated trace values
    let mut constraint_builder = CombinedPolyRowBuilder::new(num_constraints);

    let project = |x: &U::Scalar| {
        projected_scalars
            .get(x)
            .cloned()
            .expect("all scalars should have been projected at this point")
    };

    U::constrain_general(
        &mut constraint_builder,
        TraceRow::from_slice_with_layout(&up_evals, uair_sig.total_cols().as_column_layout()),
        TraceRow::from_slice_with_layout(&down_evals, down_layout),
        &project,
        |x, y| Some(project(y) * x),
        ImpossibleIdeal::from_ref,
    );

    let mut combined_evaluations = constraint_builder.combined_evaluations;
    combined_evaluations.iter_mut().for_each(|eval| eval.trim());

    Ok(combined_evaluations)
}

pub struct CombinedPolyRowBuilder<F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilder<F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, expr: Self::Expr, _ideal: &Self::Ideal) {
        self.combined_evaluations.push(expr);
    }

    fn assert_zero(&mut self, expr: Self::Expr) {
        self.combined_evaluations.push(expr);
    }
}

/// Z-only row builder: pre-fills `combined_evaluations` with `ZERO` for
/// every constraint and only writes at slots whose tag is
/// [`ConstraintRing::Z`]. Used by the dual-prime Z-branch IC together
/// with a UAIR that gates its F_p / zero-ideal sub-graphs via
/// [`ConstraintBuilder::is_active_for`] / [`ConstraintBuilder::is_active_for_zero_ideal`]
/// — those gates short-circuit before the expensive polynomial
/// arithmetic, leaving only the Z-tagged work per row.
pub struct CombinedPolyRowBuilderZOnly<'a, F: PrimeField> {
    combined_evaluations: Vec<DynamicPolynomialF<F>>,
    z_indices: &'a [usize],
    next_z_local: usize,
}

impl<'a, F: PrimeField> CombinedPolyRowBuilderZOnly<'a, F> {
    pub fn new(num_constraints: usize, z_indices: &'a [usize]) -> Self {
        Self {
            combined_evaluations: vec![DynamicPolynomialF::zero(); num_constraints],
            z_indices,
            next_z_local: 0,
        }
    }
}

impl<F: PrimeField> ConstraintBuilder for CombinedPolyRowBuilderZOnly<'_, F> {
    type Expr = DynamicPolynomialF<F>;
    type Ideal = ImpossibleIdeal;

    fn assert_in_ideal(&mut self, _expr: Self::Expr, _ideal: &Self::Ideal) {
        // Untyped asserts (default tag = Z in IdealCollector) — the
        // user's gate should prevent them firing here. Treat as no-op.
    }

    fn assert_in_ideal_typed(
        &mut self,
        expr: Self::Expr,
        _ideal: &Self::Ideal,
        ring: ConstraintRing,
    ) {
        if ring == ConstraintRing::Z && self.next_z_local < self.z_indices.len() {
            let idx = self.z_indices[self.next_z_local];
            self.combined_evaluations[idx] = expr;
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.next_z_local += 1;
            }
        }
    }

    fn assert_zero(&mut self, _expr: Self::Expr) {
        // Zero-ideal slots are not produced by the Z-branch IC.
    }

    #[inline(always)]
    fn is_active_for(&self, ring: ConstraintRing) -> bool {
        ring == ConstraintRing::Z
    }

    #[inline(always)]
    fn is_active_for_zero_ideal(&self) -> bool {
        false
    }
}

/// Z-only variant of [`compute_combined_polynomials`]. Uses
/// [`CombinedPolyRowBuilderZOnly`] so each row's `U::constrain_general`
/// call only walks the Z-tagged sub-graph (assuming the UAIR gates its
/// F_p / zero-ideal sections via [`ConstraintBuilder::is_active_for`]).
///
/// `z_indices` lists the global constraint-vector positions of
/// Z-tagged non-zero-ideal slots, in declaration order.
#[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
pub fn compute_combined_polynomials_z_only<F, U>(
    trace_matrix: &RowMajorTrace<F>,
    projected_scalars: &HashMap<U::Scalar, DynamicPolynomialF<F>>,
    num_constraints: usize,
    field_cfg: &F::Config,
    z_indices: &[usize],
) -> Vec<Vec<DenseMultilinearExtension<F::Inner>>>
where
    F: PrimeField,
    U: Uair,
{
    let field_zero = F::zero_with_cfg(field_cfg);
    let uair_sig = U::signature();
    let down_layout = uair_sig.down_cols().as_column_layout();
    let num_rows = trace_matrix.len();

    let skip: Vec<bool> = {
        let mut s = vec![true; num_constraints];
        for &i in z_indices {
            s[i] = false;
        }
        s
    };

    let mut max_degrees_and_combined_poly_rows: Vec<(usize, Vec<DynamicPolynomialF<F>>)> =
        cfg_into_iter!(0..num_rows - 1)
            .map(|row_idx| {
                let up = &trace_matrix[row_idx];

                let down: Vec<DynamicPolynomialF<F>> = uair_sig
                    .shifts()
                    .iter()
                    .map(|spec| {
                        if row_idx + spec.shift_amount() < num_rows {
                            trace_matrix[row_idx + spec.shift_amount()][spec.source_col()].clone()
                        } else {
                            DynamicPolynomialF::zero()
                        }
                    })
                    .collect();

                let mut builder = CombinedPolyRowBuilderZOnly::new(num_constraints, z_indices);

                let project = |x: &U::Scalar| {
                    projected_scalars
                        .get(x)
                        .cloned()
                        .expect("all scalars should have been projected at this point")
                };

                U::constrain_general(
                    &mut builder,
                    TraceRow::from_slice_with_layout(up, uair_sig.total_cols().as_column_layout()),
                    TraceRow::from_slice_with_layout(&down, down_layout),
                    &project,
                    |x, y| Some(project(y) * x),
                    ImpossibleIdeal::from_ref,
                );

                let mut combined_evaluations = builder.combined_evaluations;
                combined_evaluations.iter_mut().for_each(|eval| eval.trim());

                let max_degree = z_indices
                    .iter()
                    .map(|&i| combined_evaluations[i].degree().unwrap_or(0))
                    .max()
                    .unwrap_or(0);

                (max_degree, combined_evaluations)
            })
            .collect();

    let max_degree = *max_degrees_and_combined_poly_rows
        .iter()
        .map(|(max_degree, _)| max_degree)
        .max()
        .expect("Z-branch IC must have at least one row");

    max_degrees_and_combined_poly_rows.push((0, vec![DynamicPolynomialF::zero(); num_constraints]));

    prepare_coefficient_mles(
        num_constraints,
        max_degree,
        &max_degrees_and_combined_poly_rows,
        field_zero.inner(),
        &skip,
    )
}

impl<F: PrimeField> CombinedPolyRowBuilder<F> {
    pub fn new(num_constraints: usize) -> Self {
        Self {
            combined_evaluations: Vec::with_capacity(num_constraints),
        }
    }
}
