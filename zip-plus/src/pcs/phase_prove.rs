use crate::{
    ZipError,
    code::LinearCode,
    pcs::{
        structs::{ZipPlus, ZipPlusHint, ZipPlusParams, ZipTypes},
        utils::{point_to_tensor, validate_input},
    },
    pcs_transcript::PcsProverTranscript,
};
use crypto_primitives::{FromWithConfig, IntoWithConfig, PrimeField};
use itertools::Itertools;
use num_traits::{ConstOne, ConstZero, Zero};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use zinc_poly::{Polynomial, mle::DenseMultilinearExtension};
use zinc_transcript::traits::{Transcribable, Transcript};
use zinc_utils::{
    UNCHECKED, cfg_chunks, cfg_iter, cfg_iter_mut,
    from_ref::FromRef,
    inner_product::{InnerProduct, MBSInnerProduct},
    montgomery_inner_product::MontgomeryIntegerInnerProduct,
    mul_by_scalar::MulByScalar,
};

impl<Zt: ZipTypes, Lc: LinearCode<Zt>> ZipPlus<Zt, Lc> {
    /// Generates an opening proof for one or more committed multilinear
    /// polynomials at an evaluation point, using the Zip+ protocol.
    ///
    /// This replaces the old two-phase (test + evaluate) approach with a single
    /// merged phase. The key idea: alpha-projection (Eval → CombR) is used for
    /// *both* the proximity argument and the evaluation claim, eliminating the
    /// separate field-domain projection via `projecting_element` γ.
    ///
    /// # Algorithm
    /// 1. Computes points: `(q_0, q_1) = point_to_tensor(point)` where `q_0`
    ///    (length `num_rows`) combines rows and `q_1` (length `row_len`)
    ///    combines columns.
    /// 2. Per polynomial, samples random challenges `alphas` (`[α_0, …, α_d]`).
    ///    For each decoded row `w_j` takes the inner product `<entry, alphas>`
    ///    of every entry in the row, producing `w'_j` — a row of `CombR`
    ///    integers.
    /// 3. Computes `b` (length `num_rows`), accumulated across all polys: `b_j
    ///    += <w'_j, q_1>` for each row `j`.
    /// 4. Writes `b` to the transcript and computes `eval = <q_0, b>`.
    /// 5. Samples combination coefficients `betas` (or hardcodes `[1]` when
    ///    `num_rows == 1`) and computes `combined_row` (CombR, length
    ///    `row_len`) = `sum_i(sum_j(s_j * w'_ij))`, accumulated across all
    ///    polynomials
    /// 6. Writes `combined_row` to the transcript.
    /// 7. Opens `NUM_COLUMN_OPENINGS` Merkle columns: for each, squeezes a
    ///    column index, writes per-polynomial column values (Cw entries), and
    ///    appends the Merkle proof.
    ///
    /// # Transcript layout
    /// ```text
    /// [field_cfg sampled]
    /// [per-poly alphas sampled]
    /// [b written as F elements]
    /// [coeffs s sampled (or hardcoded [1])]
    /// [combined_row written as CombR]
    /// [column openings: idx, per-poly column values, merkle proof] × NUM_COLUMN_OPENINGS
    /// ```
    ///
    /// # Parameters
    /// - `pp`: Public parameters containing `num_vars`, `num_rows`, and the
    ///   linear code configuration.
    /// - `polys`: Slice of multilinear polynomials (batch). All must have
    ///   `num_vars` variables matching `pp`.
    /// - `point`: The evaluation point (in `Zt::Pt` coordinates, length
    ///   `num_vars`).
    /// - `commit_hint`: The `ZipPlusHint` returned by `commit`, containing
    ///   per-polynomial codeword matrices and the shared Merkle tree.
    ///
    /// # Returns
    /// A `Result` containing:
    /// - `F`: The combined evaluation `<q_0, b>`, which equals
    ///   `sum_i(alpha_projected_eval_i(point))` across all batched polys.
    /// - `ZipPlusProof`: The serialized transcript (b, combined_row, column
    ///   openings + Merkle proofs) for the verifier.
    ///
    /// # Errors
    /// - Returns `ZipError::InvalidPcsParam` if any polynomial has more
    ///   variables than `pp` supports.
    /// - Returns `ZipError::OverflowError` (when `CHECK_FOR_OVERFLOW` is true)
    ///   if intermediate CombR sums exceed the integer precision.
    pub fn prove<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[Zt::Pt],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>
            + MontgomeryIntegerInnerProduct<Zt::CombR>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let point = point
            .iter()
            .map(|v| v.into_with_cfg(field_cfg))
            .collect::<Vec<F>>();
        Self::prove_f::<F, CHECK_FOR_OVERFLOW>(
            transcript,
            pp,
            polys,
            &point,
            commit_hint,
            field_cfg,
        )
    }

    /// See [`Self::prove`] for details.
    /// This version takes the evaluation point already mapped to the field
    #[allow(clippy::arithmetic_side_effects)]
    pub fn prove_f<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        polys: &[DenseMultilinearExtension<Zt::Eval>],
        point: &[F],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>
            + MontgomeryIntegerInnerProduct<Zt::CombR>,
        F::Inner: Transcribable,
        F::Modulus: Transcribable,
    {
        let batch_size = polys.len();
        validate_input::<Zt, Lc, _>(
            "prove",
            pp.num_vars,
            pp.linear_code.row_len(),
            batch_size,
            polys,
            &[point],
        )?;

        let num_rows = pp.num_rows;
        let row_len = pp.linear_code.row_len();

        let (q_0, q_1) = point_to_tensor(num_rows, point, field_cfg)?;

        let degree_bound = Zt::Comb::DEGREE_BOUND;
        let polys_as_comb_r: Vec<Vec<Zt::CombR>> = polys
            .iter()
            .map(|poly| {
                let alphas = if degree_bound.is_zero() {
                    vec![Zt::Chal::ONE]
                } else {
                    transcript.fs_transcript.get_challenges(degree_bound + 1)
                };

                cfg_iter!(poly.evaluations)
                    .map(|eval| {
                        Zt::EvalDotChal::inner_product::<CHECK_FOR_OVERFLOW>(
                            eval,
                            &alphas,
                            Zt::CombR::ZERO,
                        )
                        .map_err(ZipError::from)
                    })
                    .collect()
            })
            .try_collect()?;

        let zero_f = F::zero_with_cfg(field_cfg);
        let q_1_montgomery =
            <F as MontgomeryIntegerInnerProduct<Zt::CombR>>::prepare_montgomery_rhs(&q_1, &zero_f)?;

        // Compute per-polynomial row dot products, then sum across polynomials.
        let b = if batch_size == 1 {
            cfg_chunks!(&polys_as_comb_r[0], row_len)
                .map(|row| {
                    <F as MontgomeryIntegerInnerProduct<
                        Zt::CombR,
                    >>::inner_product_prepared_montgomery(
                        row,
                        &q_1_montgomery,
                    )
                })
                .collect::<Result<Vec<F>, _>>()?
        } else {
            let per_poly_b: Vec<Vec<F>> = cfg_iter!(polys_as_comb_r)
                .map(|poly_comb_r| {
                    cfg_chunks!(poly_comb_r, row_len)
                        .map(|row| {
                            <F as MontgomeryIntegerInnerProduct<
                                Zt::CombR,
                            >>::inner_product_prepared_montgomery(
                                row,
                                &q_1_montgomery,
                            )
                        })
                        .collect::<Result<Vec<F>, _>>()
                })
                .collect::<Result<_, _>>()?;

            let mut b = vec![zero_f.clone(); num_rows];
            for poly_b in &per_poly_b {
                b.iter_mut().zip(poly_b).for_each(|(a, d)| *a += d);
            }
            b
        };

        transcript.write_field_elements(&b)?;
        // Compute eval = <q_0, b> (inner product in field), <q_2, b> in paper
        // It is safe to use inner_product_unchecked because we're in a field.
        let eval = MBSInnerProduct::inner_product::<UNCHECKED>(&q_0, &b, zero_f.clone())?;

        // Matrix-vector product over the flat poly_comb_r layout:
        // Each poly is a row-major (num_rows x row_len) matrix, and coeffs is the
        // vector.
        // combined_row[col] = sum_i sum_j (coeffs[j] * poly_i[j * row_len + col])

        let coeffs = if pp.num_rows == 1 {
            vec![Zt::Chal::ONE]
        } else {
            transcript
                .fs_transcript
                .get_challenges::<Zt::Chal>(num_rows)
        };

        let combined_row: Vec<Zt::CombR> = {
            let mut combined = vec![Zt::CombR::ZERO; row_len];
            cfg_iter_mut!(combined).enumerate().try_for_each(
                |(col, acc)| -> Result<(), ZipError> {
                    for poly_comb_r in &polys_as_comb_r {
                        // Strided access: skip to column `col`, then step by `row_len`
                        // to pick the col-th entry of each logical row.
                        for (eval, coeff) in poly_comb_r
                            .iter()
                            .skip(col)
                            .step_by(row_len)
                            .zip(coeffs.iter())
                        {
                            let scaled: Zt::CombR = eval
                                .mul_by_scalar::<CHECK_FOR_OVERFLOW>(coeff)
                                .expect("Cannot multiply evaluation by coefficient");
                            if CHECK_FOR_OVERFLOW {
                                *acc = zinc_utils::add!(
                                    *acc,
                                    &scaled,
                                    "Addition overflow while combining rows across polys"
                                );
                            } else {
                                *acc += scaled;
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            combined
        };

        transcript.write_const_many(&combined_row)?;
        for _ in 0..Zt::NUM_COLUMN_OPENINGS {
            let column_idx = transcript.squeeze_challenge_idx(pp.linear_code.codeword_len());
            Self::open_merkle_trees_for_column(transcript, commit_hint, column_idx)?;
        }

        Ok(eval)
    }

    /// See [`Self::prove`] for details.
    #[inline(always)]
    pub fn prove_single<F, const CHECK_FOR_OVERFLOW: bool>(
        transcript: &mut PcsProverTranscript,
        pp: &ZipPlusParams<Zt, Lc>,
        poly: &DenseMultilinearExtension<Zt::Eval>,
        point: &[Zt::Pt],
        commit_hint: &ZipPlusHint<Zt::Cw>,
        field_cfg: &F::Config,
    ) -> Result<F, ZipError>
    where
        F: PrimeField
            + for<'a> FromWithConfig<&'a Zt::Chal>
            + for<'a> FromWithConfig<&'a Zt::Pt>
            + for<'a> MulByScalar<&'a F>
            + FromRef<F>
            + MontgomeryIntegerInnerProduct<Zt::CombR>,
        F::Inner: Transcribable,
        F::Modulus: FromRef<Zt::Fmod> + Transcribable,
    {
        Self::prove::<F, CHECK_FOR_OVERFLOW>(
            transcript,
            pp,
            std::slice::from_ref(poly),
            point,
            commit_hint,
            field_cfg,
        )
    }

    pub(super) fn open_merkle_trees_for_column(
        transcript: &mut PcsProverTranscript,
        commit_hint: &ZipPlusHint<Zt::Cw>,
        column_idx: usize,
    ) -> Result<(), ZipError> {
        for cw_matrix in &commit_hint.cw_matrices {
            let column_values = cw_matrix.as_rows().map(|row| &row[column_idx]);
            transcript.write_const_many_iter(column_values, cw_matrix.num_rows)?;
        }

        let merkle_proof = commit_hint
            .merkle_tree
            .prove(column_idx)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to open merkle tree".into()))?;
        transcript
            .write_merkle_proof(&merkle_proof)
            .map_err(|_| ZipError::InvalidPcsOpen("Failed to write a merkle tree proof".into()))?;

        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]
mod tests {
    use crate::{
        code::{LinearCode, iprs::IprsCode},
        merkle::MerkleTree,
        pcs::{
            structs::{ZipPlus, ZipPlusHint, ZipTypes},
            test_utils::*,
        },
        pcs_transcript::PcsProverTranscript,
    };
    use crypto_bigint::{U64, Word};
    use crypto_primitives::{
        IntoWithConfig, crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_int::Int,
        crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
    };
    use num_traits::{ConstOne, Zero};
    use zinc_poly::mle::DenseMultilinearExtension;
    use zinc_primality::MillerRabin;
    use zinc_transcript::traits::Transcript;
    use zinc_utils::{
        CHECKED,
        from_ref::FromRef,
        inner_product::{MBSInnerProduct, ScalarProduct},
    };

    const INT_LIMBS: usize = U64::LIMBS;

    const N: usize = INT_LIMBS;
    const K: usize = INT_LIMBS * 4;
    const M: usize = INT_LIMBS * 8;
    const DEGREE_PLUS_ONE: usize = 3;

    type F = BoxedMontyField;

    type Zt = TestZipTypes<N, K, M>;
    type C = IprsCode<Zt, TestIprsConfig, REP_FACTOR, CHECKED>;

    type PolyZt = TestBinPolyZipTypes<K, M, DEGREE_PLUS_ONE>;
    type PolyC = IprsCode<PolyZt, TestIprsConfig, REP_FACTOR, CHECKED>;

    type TestZip = ZipPlus<Zt, C>;
    type TestPolyZip = ZipPlus<PolyZt, PolyC>;

    #[derive(Debug, Clone)]
    struct WideSignedZipTypes;

    impl ZipTypes for WideSignedZipTypes {
        const NUM_COLUMN_OPENINGS: usize = 8;
        type Eval = Int<WIDE_EVAL_LIMBS>;
        type Cw = Int<WIDE_EVAL_LIMBS>;
        type Fmod = Uint<WIDE_FIELD_LIMBS>;
        type PrimeTest = MillerRabin;
        type Chal = Int<WIDE_FIELD_LIMBS>;
        type Pt = Int<WIDE_FIELD_LIMBS>;
        type CombR = Int<WIDE_COMB_LIMBS>;
        type Comb = Self::CombR;
        type EvalDotChal = ScalarProduct;
        type CombDotChal = ScalarProduct;
        type ArrCombRDotChal = MBSInnerProduct;
    }

    const WIDE_FIELD_LIMBS: usize = INT_LIMBS;
    const WIDE_EVAL_LIMBS: usize = INT_LIMBS * 2;
    const WIDE_COMB_LIMBS: usize = INT_LIMBS * 4;

    type WideF = MontyField<WIDE_FIELD_LIMBS>;
    type WideC = IprsCode<WideSignedZipTypes, TestIprsConfig, REP_FACTOR, CHECKED>;
    type WideZip = ZipPlus<WideSignedZipTypes, WideC>;

    fn test_point(num_vars: usize) -> Vec<Int<INT_LIMBS>> {
        (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect()
    }

    fn wide_signed_eval(i: usize) -> Int<WIDE_EVAL_LIMBS> {
        let mut words = [0; WIDE_EVAL_LIMBS];
        words[0] = (i as Word).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        words[1] = if i.is_multiple_of(3) { 1 } else { 0 };
        let value = Int::from_words(words);
        if i.is_multiple_of(2) { -value } else { value }
    }

    #[test]
    fn prove_handles_dense_q1_with_wide_signed_coefficients() {
        // The optimized prover path must match the verifier's field-lift
        // semantics when q_1 is dense and CombR is wider than the field.
        let num_vars = 9;
        let poly_size = 1 << num_vars;
        let pp = WideZip::setup(
            poly_size,
            WideC::new(IPRS_ROW_LEN, IPRS_DEPTH).expect("valid IPRS parameters"),
        );

        assert_eq!(pp.linear_code.row_len(), 1 << (num_vars - 1));
        assert_eq!(pp.num_rows, 2);

        let poly = DenseMultilinearExtension {
            num_vars,
            evaluations: (0..poly_size).map(wide_signed_eval).collect(),
        };
        let (hint, comm) = WideZip::commit_single(&pp, &poly).expect("commit should succeed");

        let point: Vec<<WideSignedZipTypes as ZipTypes>::Pt> =
            (0..num_vars).map(|i| Int::from(i as i32 + 2)).collect();

        let mut prover_transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg =
            get_field_cfg::<WideSignedZipTypes, WideF>(&mut prover_transcript.fs_transcript);

        let eval_f = WideZip::prove_single::<WideF, CHECKED>(
            &mut prover_transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        )
        .expect("wide signed prove should succeed");

        let point_f: Vec<WideF> = point.iter().map(|v| v.into_with_cfg(&field_cfg)).collect();

        let mut verifier_transcript = prover_transcript.into_verification_transcript();
        verifier_transcript.fs_transcript.absorb_slice(&comm.root);
        let field_cfg =
            get_field_cfg::<WideSignedZipTypes, WideF>(&mut verifier_transcript.fs_transcript);

        let result = WideZip::verify::<WideF, CHECKED>(
            &mut verifier_transcript,
            &pp,
            &comm,
            &field_cfg,
            &point_f,
            &eval_f,
        );

        assert!(
            result.is_ok(),
            "wide signed dense-q1 verification failed: {result:?}"
        );
    }

    #[test]
    fn prove_succeeds_for_single_poly() {
        let num_vars = 10;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_succeeds_for_poly_type() {
        let num_vars = 10;
        let (pp, poly) = setup_poly_test_params(num_vars);
        let (hint, comm) = TestPolyZip::commit_single(&pp, &poly).unwrap();
        let point: Vec<i128> = (0..num_vars).map(|i| i as i128 + 2).collect();

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestPolyZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_succeeds_with_corrupted_codeword() {
        let num_vars = 10;
        let (pp, poly) = setup_test_params(num_vars);
        let (mut hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();

        {
            let mut rows = hint.cw_matrices[0].to_rows_slices_mut();
            assert!(!rows.is_empty());
            rows[0][0] += Int::ONE;
        }

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &corrupted_hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial() {
        let num_vars = 10;
        let (pp, _) = setup_test_params(num_vars);
        let oversized_poly: DenseMultilinearExtension<_> =
            (0..1 << (num_vars + 1)).map(Int::from).collect();

        let (hint, comm) =
            TestZip::commit_single(&pp, &setup_test_params::<N, K, M>(num_vars).1).unwrap();

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &oversized_poly,
            &point,
            &hint,
            &field_cfg,
        );
        assert!(result.is_err());
    }

    /// For TestZipTypes (degree_bound = 0), alphas = [1] so prove eval
    /// equals poly(point) lifted to F.
    #[test]
    fn prove_returns_correct_evaluation() {
        let num_vars = 10;
        let (pp, poly) = setup_test_params(num_vars);
        let (hint, comm) = TestZip::commit_single(&pp, &poly).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let eval_f = TestZip::prove_single::<F, CHECKED>(
            &mut transcript,
            &pp,
            &poly,
            &point,
            &hint,
            &field_cfg,
        )
        .unwrap();

        let poly_wide: DenseMultilinearExtension<Int<M>> =
            poly.evaluations.iter().map(Int::from_ref).collect();
        let expected_int = poly_wide.evaluate(&point, Zero::zero()).unwrap();
        let expected_f: F = (&expected_int).into_with_cfg(&field_cfg);

        assert_eq!(eval_f, expected_f);
    }

    fn make_batch_polys(
        num_vars: usize,
        batch_size: usize,
    ) -> Vec<DenseMultilinearExtension<Int<INT_LIMBS>>> {
        let poly_size = 1 << num_vars;
        (0..batch_size)
            .map(|b| {
                let base = (b * poly_size) as i32;
                (base + 1..=base + poly_size as i32)
                    .map(Int::from)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn prove_succeeds_for_batch() {
        let num_vars = 10;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 2);

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_succeeds_for_batch_5() {
        let num_vars = 10;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 5);

        let (hint, comm) = TestZip::commit(&pp, &polys).unwrap();
        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_ok())
    }

    #[test]
    fn prove_with_corrupted_codeword_for_batch() {
        let num_vars = 10;
        let (pp, _) = setup_test_params(num_vars);
        let polys = make_batch_polys(num_vars, 2);

        let (mut hint, comm) = TestZip::commit(&pp, &polys).unwrap();

        hint.cw_matrices[0].to_rows_slices_mut()[0][0] += Int::ONE;

        let corrupted_tree = {
            let all_rows: Vec<&[_]> = hint.cw_matrices.iter().flat_map(|m| m.as_rows()).collect();
            MerkleTree::new(&all_rows)
        };
        let corrupted_hint = ZipPlusHint::new(hint.cw_matrices, corrupted_tree);

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result = TestZip::prove::<F, CHECKED>(
            &mut transcript,
            &pp,
            &polys,
            &point,
            &corrupted_hint,
            &field_cfg,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn prove_rejects_oversized_polynomial_in_batch() {
        let num_vars = 10;
        let (pp, _) = setup_test_params(num_vars);
        let oversized: DenseMultilinearExtension<_> = (0..1 << 5).map(Int::from).collect();
        let normal: DenseMultilinearExtension<_> = (1..=16).map(Int::from).collect();
        let polys = vec![normal, oversized];

        let (hint, comm) = TestZip::commit(&pp, &make_batch_polys(num_vars, 2)).unwrap();

        let point = test_point(num_vars);

        let mut transcript = PcsProverTranscript::new_from_commitment(&comm);
        let field_cfg = get_field_cfg::<Zt, F>(&mut transcript.fs_transcript);

        let result =
            TestZip::prove::<F, CHECKED>(&mut transcript, &pp, &polys, &point, &hint, &field_cfg);
        assert!(result.is_err());
    }
}
