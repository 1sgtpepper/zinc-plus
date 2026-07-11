use crypto_bigint::{NonZero, Uint as CBUint};
use crypto_primitives::{
    Field, FromWithConfig, HasPrimeFieldConfig, IntRing, PrimeField,
    crypto_bigint_boxed_monty::BoxedMontyField, crypto_bigint_boxed_uint::BoxedUint,
    crypto_bigint_const_monty::ConstMontyField, crypto_bigint_int::Int,
    crypto_bigint_monty::MontyField, crypto_bigint_uint::Uint,
};
use std::ops::{AddAssign, Mul, SubAssign};

use thiserror::Error;

/// Error type for Montgomery-prepared inner product operations.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum MontgomeryError {
    #[error("The length of LHS and RHS does not match: LHS={lhs}, RHS={rhs}")]
    LengthMismatch { lhs: usize, rhs: usize },
    #[error("The field configuration does not match")]
    FieldConfigMismatch,
    #[error("The coefficient does not fit the field precision")]
    CoefficientOutOfRange,
}

/// Right-hand side field elements prepared for repeated Montgomery dot
/// products.
pub struct PreparedMontgomeryRhs<F: PrimeField> {
    // Values are shifted by one extra Montgomery factor so that multiplying by
    // lhs coefficients injected with `new_unchecked_with_cfg` produces an
    // ordinary field product.
    shifted_values: Vec<F>,
    cfg: F::Config,
}

/// Computes the same field result as `<lhs, rhs>` after mapping each signed
/// integer coefficient into the field, while avoiding full Montgomery
/// conversion for every lhs entry.
pub trait MontgomeryIntegerInnerProduct<Lhs>: PrimeField {
    type PreparedRhs: Sync;

    fn prepare_montgomery_rhs(
        rhs: &[Self],
        zero: &Self,
    ) -> Result<Self::PreparedRhs, MontgomeryError>;

    fn inner_product_prepared_montgomery(
        lhs: &[Lhs],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, MontgomeryError>;
}

fn abs_as_field_width<const FIELD_LIMBS: usize, const INT_LIMBS: usize>(
    value: &Int<INT_LIMBS>,
    modulus: &NonZero<CBUint<FIELD_LIMBS>>,
) -> CBUint<FIELD_LIMBS> {
    let abs = value.inner().abs();
    if FIELD_LIMBS < INT_LIMBS {
        if abs.as_words()[FIELD_LIMBS..].iter().all(|word| *word == 0) {
            let abs = abs.resize();
            if abs < *modulus.as_ref() {
                return abs;
            }
        }
        let wide_modulus = NonZero::new(modulus.as_ref().resize::<INT_LIMBS>()).unwrap();
        abs.rem(&wide_modulus).resize()
    } else {
        let abs = abs.resize();
        if abs >= *modulus.as_ref() {
            abs.rem(modulus)
        } else {
            abs
        }
    }
}

#[allow(clippy::arithmetic_side_effects)]
fn inner_product_prepared_with_abs<F, const INT_LIMBS: usize>(
    lhs: &[Int<INT_LIMBS>],
    rhs: &PreparedMontgomeryRhs<F>,
    mut abs_to_inner: impl FnMut(&Int<INT_LIMBS>, &F::Config) -> Result<F::Inner, MontgomeryError>,
) -> Result<F, MontgomeryError>
where
    F: PrimeField
        + for<'a> AddAssign<&'a F>
        + for<'a> Mul<&'a F, Output = F>
        + for<'a> SubAssign<&'a F>,
{
    if lhs.len() != rhs.shifted_values.len() {
        return Err(MontgomeryError::LengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.shifted_values.len(),
        });
    }

    let cfg = rhs.cfg.clone();
    let mut acc = F::zero_with_cfg(&cfg);

    for (coeff, q) in lhs.iter().zip(&rhs.shifted_values) {
        let term = F::new_unchecked_with_cfg(abs_to_inner(coeff, &cfg)?, &cfg) * q;
        if coeff.is_negative() {
            acc -= &term;
        } else {
            acc += &term;
        }
    }

    Ok(acc)
}

impl<const FIELD_LIMBS: usize, const INT_LIMBS: usize> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>
    for MontyField<FIELD_LIMBS>
{
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(
        rhs: &[Self],
        zero: &Self,
    ) -> Result<Self::PreparedRhs, MontgomeryError> {
        let cfg = *zero.cfg();
        let shifted_values = rhs
            .iter()
            .map(|q| -> Result<_, MontgomeryError> {
                if q.cfg().modulus() != cfg.modulus() {
                    return Err(MontgomeryError::FieldConfigMismatch);
                }
                Ok(Self::from_with_cfg(q.inner(), &cfg))
            })
            .collect::<Result<_, _>>()?;
        Ok(PreparedMontgomeryRhs {
            shifted_values,
            cfg,
        })
    }

    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, MontgomeryError> {
        inner_product_prepared_with_abs(lhs, rhs, |coeff, cfg| {
            Ok(Uint::new(abs_as_field_width(
                coeff,
                cfg.modulus().as_nz_ref(),
            )))
        })
    }
}

impl<
    Mod: crypto_bigint::modular::ConstMontyParams<FIELD_LIMBS>,
    const FIELD_LIMBS: usize,
    const INT_LIMBS: usize,
> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>> for ConstMontyField<Mod, FIELD_LIMBS>
{
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(
        rhs: &[Self],
        _zero: &Self,
    ) -> Result<Self::PreparedRhs, MontgomeryError> {
        let shifted_values = rhs.iter().map(|q| Self::from(*q.inner())).collect();
        Ok(PreparedMontgomeryRhs {
            shifted_values,
            cfg: (),
        })
    }

    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, MontgomeryError> {
        inner_product_prepared_with_abs(lhs, rhs, |coeff, _| {
            if INT_LIMBS > FIELD_LIMBS {
                return Err(MontgomeryError::CoefficientOutOfRange);
            }
            Ok(Uint::new(abs_as_field_width(
                coeff,
                Mod::PARAMS.modulus().as_nz_ref(),
            )))
        })
    }
}

impl<const INT_LIMBS: usize> MontgomeryIntegerInnerProduct<Int<INT_LIMBS>> for BoxedMontyField {
    type PreparedRhs = PreparedMontgomeryRhs<Self>;

    fn prepare_montgomery_rhs(
        rhs: &[Self],
        zero: &Self,
    ) -> Result<Self::PreparedRhs, MontgomeryError> {
        let cfg = zero.cfg().clone();
        let shifted_values = rhs
            .iter()
            .map(|q| -> Result<_, MontgomeryError> {
                if q.cfg().modulus() != cfg.modulus()
                    || q.cfg().modulus().bits_precision() != cfg.modulus().bits_precision()
                {
                    return Err(MontgomeryError::FieldConfigMismatch);
                }
                Ok(Self::from_with_cfg(q.inner(), &cfg))
            })
            .collect::<Result<_, _>>()?;
        Ok(PreparedMontgomeryRhs {
            shifted_values,
            cfg,
        })
    }

    fn inner_product_prepared_montgomery(
        lhs: &[Int<INT_LIMBS>],
        rhs: &Self::PreparedRhs,
    ) -> Result<Self, MontgomeryError> {
        inner_product_prepared_with_abs(lhs, rhs, |coeff, cfg| {
            let abs: BoxedUint = coeff.inner().abs().into();
            abs.try_resize(cfg.modulus().bits_precision())
                .ok_or(MontgomeryError::CoefficientOutOfRange)
        })
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use crypto_bigint::{U64, Word, const_monty_params};
    use crypto_primitives::{
        ConstSemiring, FromWithConfig, crypto_bigint_boxed_monty::BoxedMontyField,
    };
    use proptest::prelude::*;

    use crate::inner_product::MBSInnerProduct;

    const FIELD_LIMBS: usize = U64::LIMBS;
    const INT_LIMBS: usize = U64::LIMBS * 2;
    const_monty_params!(Params7, U64, "0000000000000007");

    type FixedF = MontyField<FIELD_LIMBS>;
    type BoxedF = BoxedMontyField;
    type ConstF = ConstMontyField<Params7, FIELD_LIMBS>;

    fn large_positive() -> Int<INT_LIMBS> {
        Int::from_words([0, 1_u64.wrapping_shl(16) as Word])
    }

    fn fixed_cfg() -> <FixedF as HasPrimeFieldConfig>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from(7_u8))).expect("odd modulus")
    }

    fn boxed_cfg() -> <BoxedF as HasPrimeFieldConfig>::Config {
        BoxedF::make_cfg(&BoxedUint::from(7_u8)).expect("odd modulus")
    }

    fn fixed_cfg_11() -> <FixedF as HasPrimeFieldConfig>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from(11_u8))).expect("odd modulus")
    }

    fn fixed_high_word_cfg() -> <FixedF as HasPrimeFieldConfig>::Config {
        FixedF::make_cfg(&Uint::new(CBUint::from_words([0xffff_ffff_ffff_ffc5])))
            .expect("odd modulus")
    }

    fn boxed_cfg_11() -> <BoxedF as HasPrimeFieldConfig>::Config {
        BoxedF::make_cfg(&BoxedUint::from(11_u8)).expect("odd modulus")
    }

    fn boxed_wide_cfg() -> <BoxedF as HasPrimeFieldConfig>::Config {
        BoxedF::make_cfg(&BoxedUint::from((1_u128 << 127) - 1)).expect("odd modulus")
    }

    fn coeffs() -> [Int<INT_LIMBS>; 3] {
        [large_positive(), Int::from(-5_i32), Int::from(11_i32)]
    }

    fn dense_coeffs() -> Vec<Int<INT_LIMBS>> {
        (0_u64..64)
            .map(|i| {
                let mut words = [0; INT_LIMBS];
                words[0] = i * 37 + 5;
                words[1] = if i % 5 == 0 { 1 << 20 } else { 0 };
                let value = Int::from_words(words);
                if i % 2 == 0 { -value } else { value }
            })
            .collect()
    }

    fn rhs_values(len: usize) -> Vec<u64> {
        (0..len).map(|i| (i as u64 * 11 + 2) % 7).collect()
    }

    fn signed_int(lo: Word, hi: Word, is_negative: bool) -> Int<INT_LIMBS> {
        let value = Int::from_words([lo, hi]);
        if is_negative { -value } else { value }
    }

    fn manual_mod7<const LIMBS: usize>(coeffs: &[Int<LIMBS>], rhs: &[u64]) -> u64 {
        let modulus = NonZero::new(CBUint::<LIMBS>::from(7_u8)).unwrap();
        let mut acc = 0_i128;

        for (coeff, rhs) in coeffs.iter().zip(rhs) {
            let abs_mod = i128::from(coeff.inner().abs().rem(&modulus).as_words()[0]);
            let term = (abs_mod * i128::from(*rhs)) % 7;
            if coeff.is_negative() {
                acc -= term;
            } else {
                acc += term;
            }
        }

        u64::try_from(acc.rem_euclid(7)).unwrap()
    }

    #[test]
    fn fixed_montgomery_inner_product_matches_manual_modular_sum() {
        let cfg = fixed_cfg();
        let rhs = [2_u64, 3, 4].map(|x| FixedF::from_with_cfg(x, &cfg));
        let zero = FixedF::zero_with_cfg(&cfg);
        let expected = FixedF::from_with_cfg(2_u64, &cfg);
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let actual = FixedF::inner_product_prepared_montgomery(&coeffs(), &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn fixed_prepared_inner_product_matches_field_lift_for_dense_signed_coefficients() {
        let cfg = fixed_cfg();
        let coeffs = dense_coeffs();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values
            .iter()
            .map(|x| FixedF::from_with_cfg(*x, &cfg))
            .collect();
        let zero = FixedF::zero_with_cfg(&cfg);
        let expected =
            MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone()).unwrap();
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            FixedF::from_with_cfg(manual_mod7(&coeffs, &rhs_values), &cfg)
        );
    }

    #[test]
    fn fixed_prepared_inner_product_matches_field_lift_for_high_word_modulus() {
        let cfg = fixed_high_word_cfg();
        let zero = FixedF::zero_with_cfg(&cfg);
        let coeffs = [
            Int::from_words([0xffff_ffff_ffff_ff80, 0]),
            -Int::from_words([0xdead_beef_cafe_babe, 1]),
            Int::from_words([0x8000_0000_0000_0041, 2]),
            -Int::from_words([0x1234_5678_9abc_def0, 0]),
        ];
        let rhs_values: [Word; 4] = [
            0xffff_ffff_ffff_ffc4,
            0xdead_beef_cafe_babe,
            0x8000_0000_0000_0041,
            0x1234_5678_9abc_def0,
        ];
        let rhs = rhs_values.map(|value| {
            FixedF::from_with_cfg(Uint::<FIELD_LIMBS>::new(CBUint::from(value)), &cfg)
        });
        let expected =
            MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone()).unwrap();
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn boxed_montgomery_inner_product_rejects_coefficients_wider_than_field() {
        let cfg = boxed_cfg();
        let rhs = [2_u64, 3, 4].map(|x| BoxedF::from_with_cfg(x, &cfg));
        let zero = BoxedF::zero_with_cfg(&cfg);
        let prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let result = BoxedF::inner_product_prepared_montgomery(&coeffs(), &prepared);

        assert_eq!(result, Err(MontgomeryError::CoefficientOutOfRange));
    }

    #[test]
    fn boxed_prepared_inner_product_matches_field_lift_for_dense_signed_coefficients() {
        let cfg = boxed_cfg();
        let coeffs: Vec<_> = (0_i32..64)
            .map(|value| {
                let value = Int::<INT_LIMBS>::from(value * 37 + 5);
                if value.inner().as_words()[0] % 2 == 0 {
                    -value
                } else {
                    value
                }
            })
            .collect();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values
            .iter()
            .map(|x| BoxedF::from_with_cfg(*x, &cfg))
            .collect();
        let zero = BoxedF::zero_with_cfg(&cfg);
        let expected =
            MBSInnerProduct::inner_product_field::<_, BoxedF>(&coeffs, &rhs, zero.clone()).unwrap();
        let prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let actual = BoxedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn const_montgomery_inner_product_matches_manual_modular_sum() {
        let coeffs = [
            Int::<FIELD_LIMBS>::from(5_i32),
            Int::from(-5_i32),
            Int::from(11_i32),
        ];
        let rhs = [2_u64, 3, 4].map(ConstF::from);
        let zero = ConstF::zero_with_cfg(&());
        let expected = MBSInnerProduct::inner_product_field(&coeffs, &rhs, zero).unwrap();
        let prepared =
            <ConstF as MontgomeryIntegerInnerProduct<Int<FIELD_LIMBS>>>::prepare_montgomery_rhs(
                &rhs, &zero,
            )
            .unwrap();

        let actual = ConstF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn const_prepared_inner_product_matches_manual_modular_sum_for_dense_signed_coefficients() {
        let coeffs: Vec<Int<FIELD_LIMBS>> = (0_i32..64)
            .map(|value| {
                let value = Int::from(value * 37 + 5);
                if value.inner().as_words()[0] % 2 == 0 {
                    -value
                } else {
                    value
                }
            })
            .collect();
        let rhs_values = rhs_values(coeffs.len());
        let rhs: Vec<_> = rhs_values.iter().copied().map(ConstF::from).collect();
        let expected = ConstF::from(manual_mod7(&coeffs, &rhs_values));
        let prepared =
            <ConstF as MontgomeryIntegerInnerProduct<Int<FIELD_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &ConstF::zero_with_cfg(&()),
            )
            .unwrap();

        let actual = ConstF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn const_montgomery_inner_product_rejects_wider_coefficient_type() {
        let rhs = [ConstF::from(1_u64)];
        let prepared =
            <ConstF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &ConstF::zero_with_cfg(&()),
            )
            .unwrap();

        let result =
            ConstF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared);

        assert_eq!(result, Err(MontgomeryError::CoefficientOutOfRange));
    }

    #[test]
    fn prepared_inner_products_match_field_lift_for_minimum_signed_coefficient() {
        let coeffs = [Int::<INT_LIMBS>::MIN];

        let fixed_cfg = fixed_cfg();
        let fixed_rhs = [FixedF::from_with_cfg(3_u64, &fixed_cfg)];
        let fixed_zero = FixedF::zero_with_cfg(&fixed_cfg);
        let fixed_expected = MBSInnerProduct::inner_product_field::<_, FixedF>(
            &coeffs,
            &fixed_rhs,
            fixed_zero.clone(),
        )
        .unwrap();
        let fixed_prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &fixed_rhs,
                &fixed_zero,
            )
            .unwrap();
        assert_eq!(
            FixedF::inner_product_prepared_montgomery(&coeffs, &fixed_prepared).unwrap(),
            fixed_expected
        );

        let const_coeffs = [Int::<FIELD_LIMBS>::MIN];
        let const_rhs = [ConstF::from(3_u64)];
        let const_zero = ConstF::zero_with_cfg(&());
        let const_expected = MBSInnerProduct::inner_product_field::<_, ConstF>(
            &const_coeffs,
            &const_rhs,
            const_zero,
        )
        .unwrap();
        let const_prepared =
            <ConstF as MontgomeryIntegerInnerProduct<Int<FIELD_LIMBS>>>::prepare_montgomery_rhs(
                &const_rhs,
                &const_zero,
            )
            .unwrap();
        assert_eq!(
            ConstF::inner_product_prepared_montgomery(&const_coeffs, &const_prepared).unwrap(),
            const_expected
        );

        let boxed_cfg = boxed_wide_cfg();
        let boxed_rhs = [BoxedF::from_with_cfg(3_u64, &boxed_cfg)];
        let boxed_zero = BoxedF::zero_with_cfg(&boxed_cfg);
        let boxed_expected = MBSInnerProduct::inner_product_field::<_, BoxedF>(
            &coeffs,
            &boxed_rhs,
            boxed_zero.clone(),
        )
        .unwrap();
        let boxed_prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &boxed_rhs,
                &boxed_zero,
            )
            .unwrap();
        assert_eq!(
            BoxedF::inner_product_prepared_montgomery(&coeffs, &boxed_prepared).unwrap(),
            boxed_expected
        );
    }

    #[test]
    fn montgomery_inner_product_rejects_length_mismatch() {
        let cfg = fixed_cfg();
        let zero = FixedF::zero_with_cfg(&cfg);
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &[],
                &zero,
            )
            .unwrap();
        let err =
            FixedF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared)
                .unwrap_err();

        assert_eq!(err, MontgomeryError::LengthMismatch { lhs: 1, rhs: 0 });
    }

    #[test]
    fn fixed_prepare_rejects_mismatched_dynamic_field_config() {
        let cfg = fixed_cfg();
        let other_cfg = fixed_cfg_11();
        let rhs = [FixedF::from_with_cfg(1_u64, &other_cfg)];

        let result =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &FixedF::zero_with_cfg(&cfg),
            );

        assert!(matches!(result, Err(MontgomeryError::FieldConfigMismatch)));
    }

    #[test]
    fn fixed_prepared_inner_product_uses_prepared_config() {
        let other_cfg = fixed_cfg_11();
        let rhs = [FixedF::from_with_cfg(1_u64, &other_cfg)];
        let prepared =
            <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &FixedF::zero_with_cfg(&other_cfg),
            )
            .unwrap();

        let actual =
            FixedF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared)
                .unwrap();

        assert_eq!(actual, FixedF::from_with_cfg(1_u64, &other_cfg));
    }

    #[test]
    fn boxed_prepare_rejects_mismatched_dynamic_field_config() {
        let cfg = boxed_cfg();
        let other_cfg = boxed_cfg_11();
        let rhs = [BoxedF::from_with_cfg(1_u64, &other_cfg)];

        let result =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &BoxedF::zero_with_cfg(&cfg),
            );

        assert!(matches!(result, Err(MontgomeryError::FieldConfigMismatch)));
    }

    #[test]
    fn boxed_prepare_rejects_same_modulus_with_different_precision() {
        let narrow_cfg = boxed_cfg();
        let wide_modulus = BoxedUint::from(7_u8).resize(128);
        let wide_cfg = BoxedF::make_cfg(&wide_modulus).expect("odd modulus");
        let rhs = [BoxedF::from_with_cfg(1_u64, &narrow_cfg)];

        let result =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &BoxedF::zero_with_cfg(&wide_cfg),
            );

        assert!(matches!(result, Err(MontgomeryError::FieldConfigMismatch)));
    }

    #[test]
    fn boxed_prepared_inner_product_uses_prepared_config() {
        let other_cfg = boxed_cfg_11();
        let rhs = [BoxedF::from_with_cfg(1_u64, &other_cfg)];
        let prepared =
            <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                &rhs,
                &BoxedF::zero_with_cfg(&other_cfg),
            )
            .unwrap();

        let actual =
            BoxedF::inner_product_prepared_montgomery(&[Int::<INT_LIMBS>::from(1_i32)], &prepared)
                .unwrap();

        assert_eq!(actual, BoxedF::from_with_cfg(1_u64, &other_cfg));
    }

    proptest! {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn fixed_prepared_inner_product_matches_field_lift_for_random_dense_inputs(
            entries in prop::collection::vec(
                (any::<Word>(), 0_u64..(1_u64 << 20), any::<bool>(), any::<Word>()),
                0..64,
            )
        ) {
            let cfg = fixed_high_word_cfg();
            let zero = FixedF::zero_with_cfg(&cfg);
            let coeffs: Vec<_> = entries
                .iter()
                .map(|(lo, hi, is_negative, _)| signed_int(*lo, *hi, *is_negative))
                .collect();
            let rhs: Vec<_> = entries
                .iter()
                .map(|(_, _, _, rhs)| {
                    FixedF::from_with_cfg(
                        Uint::<FIELD_LIMBS>::new(CBUint::from(*rhs)),
                        &cfg,
                    )
                })
                .collect();
            let expected =
                MBSInnerProduct::inner_product_field::<_, FixedF>(&coeffs, &rhs, zero.clone())
                    .unwrap();
            let prepared =
                <FixedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                    &rhs,
                    &zero,
                )
                .unwrap();

            let actual = FixedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

            prop_assert_eq!(actual, expected);
        }

        #[test]
        #[cfg_attr(miri, ignore)]
        fn boxed_prepared_inner_product_matches_field_lift_for_random_dense_inputs(
            entries in prop::collection::vec(
                (any::<Word>(), 0_u64..(1_u64 << 20), any::<bool>(), any::<Word>()),
                0..64,
            )
        ) {
            let cfg = boxed_wide_cfg();
            let zero = BoxedF::zero_with_cfg(&cfg);
            let coeffs: Vec<_> = entries
                .iter()
                .map(|(lo, hi, is_negative, _)| signed_int(*lo, *hi, *is_negative))
                .collect();
            let rhs: Vec<_> = entries
                .iter()
                .map(|(_, _, _, rhs)| BoxedF::from_with_cfg(*rhs, &cfg))
                .collect();
            let expected =
                MBSInnerProduct::inner_product_field::<_, BoxedF>(&coeffs, &rhs, zero.clone())
                    .unwrap();
            let prepared =
                <BoxedF as MontgomeryIntegerInnerProduct<Int<INT_LIMBS>>>::prepare_montgomery_rhs(
                    &rhs,
                    &zero,
                )
                .unwrap();

            let actual = BoxedF::inner_product_prepared_montgomery(&coeffs, &prepared).unwrap();

            prop_assert_eq!(actual, expected);
        }
    }
}
