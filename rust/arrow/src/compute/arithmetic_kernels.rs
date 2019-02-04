// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines basic arithmetic kernels for `PrimitiveArrays`.
//!
//! These kernels can leverage SIMD if available on your system.  Currently no runtime
//! detection is provided, you should enable the specific SIMD intrinsics using
//! `RUSTFLAGS="-C target-feature=+avx2"` for example.  See the
//! [here] (https://doc.rust-lang.org/stable/std/arch/) for more information.

use std::mem;
use std::ops::{Add, Div, Mul, Sub};
use std::slice::from_raw_parts_mut;

use num::Zero;

use crate::array::*;
use crate::array_data::ArrayDataBuilder;
use crate::bitmap::Bitmap;
use crate::buffer::{Buffer, MutableBuffer};
use crate::compute::array_ops::math_op;
use crate::datatypes;
use crate::error::{ArrowError, Result};

/// Creates a new `Option<Bitmap>` for use in binary kernels
fn update_bin_kernel_bitmap(left: &Option<Bitmap>, right: &Option<Bitmap>) -> Option<Buffer> {
    match &left {
        &None => match &right {
            &None => None,
            &Some(r) => Some(r.bits.clone()),
        },
        &Some(l) => match &right {
            &None => Some(l.bits.clone()),
            &Some(r) => Some(l.bits.clone() & r.bits.clone()),
        },
    }
}

/// Vectorized version of add operation
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn simd_bin_op<T, F>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
    op: F,
) -> Result<PrimitiveArray<T>>
where
    T: datatypes::ArrowNumericType + datatypes::ArrowSIMDType,
    T::Simd: Add<Output = T::Simd>
        + Sub<Output = T::Simd>
        + Mul<Output = T::Simd>
        + Div<Output = T::Simd>,
    F: Fn(T::Simd, T::Simd) -> T::Simd,
{
    if left.len() != right.len() {
        return Err(ArrowError::ComputeError(
            "Cannot perform math operation on arrays of different length".to_string(),
        ));
    }

    let data_builder = ArrayDataBuilder::new(T::get_data_type());

    let new_bit_buffer =
        update_bin_kernel_bitmap(&left.data().null_bitmap(), &right.data().null_bitmap());

    let lanes = T::lanes();
    let buffer_size = left.len() * mem::size_of::<T::Native>();
    let mut result = MutableBuffer::new(buffer_size).with_bitset(buffer_size, false);

    for i in (0..left.len()).step_by(lanes) {
        let simd_left = T::load(left.value_slice(i, lanes));
        let simd_right = T::load(right.value_slice(i, lanes));
        let simd_result = T::bin_op(simd_left, simd_right, &op);

        let result_slice: &mut [T::Native] = unsafe {
            from_raw_parts_mut(
                (result.data_mut().as_mut_ptr() as *mut T::Native).offset(i as isize),
                lanes,
            )
        };
        T::write(simd_result, result_slice);
    }

    let data = match new_bit_buffer {
        None => data_builder.add_buffer(result.freeze()).build(),
        Some(bit_buffer) => {
            let non_null_slots = 3;
            data_builder.add_buffer(result.freeze()).null_count(non_null_slots).null_bit_buffer(bit_buffer).build()
        }
//        Some(bit_buffer) => data_builder.add_buffer(result.freeze()).build()
    };

    Ok(PrimitiveArray::<T>::from(data))
}

/// Perform `left + right` operation on two arrays. If either left or right value is null
/// then the result is also null.
pub fn add<T>(
    left: &PrimitiveArray<T>,
    right: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>>
where
    T: datatypes::ArrowNumericType + datatypes::ArrowSIMDType,
    T::Native: Add<Output = T::Native>
        + Sub<Output = T::Native>
        + Mul<Output = T::Native>
        + Div<Output = T::Native>
        + Zero,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return simd_bin_op(&left, &right, |a, b| a + b);

    #[allow(unreachable_code)]
    math_op(left, right, |a, b| Ok(a + b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Int32Array;

    #[test]
    fn test_primitive_array_add() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let b = Int32Array::from(vec![6, 7, 8, 9, 8]);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let c = simd_bin_op(&a, &b, |x, y| x + y).unwrap();

            assert_eq!(11, c.value(0));
            assert_eq!(13, c.value(1));
            assert_eq!(15, c.value(2));
            assert_eq!(17, c.value(3));
            assert_eq!(17, c.value(4));
        }

        let d = add(&a, &b).unwrap();
        assert_eq!(11, d.value(0));
        assert_eq!(13, d.value(1));
        assert_eq!(15, d.value(2));
        assert_eq!(17, d.value(3));
        assert_eq!(17, d.value(4));
    }

    #[test]
    fn test_primitive_array_add_mismatched_length() {
        let a = Int32Array::from(vec![5, 6, 7, 8, 9]);
        let b = Int32Array::from(vec![6, 7, 8]);
        let e = add(&a, &b)
            .err()
            .expect("should have failed due to different lengths");
        assert_eq!(
            "ComputeError(\"Cannot perform math operation on arrays of different length\")",
            format!("{:?}", e)
        );
    }

    #[test]
    fn test_primitive_array_add_with_nulls() {
        let a = Int32Array::from(vec![Some(5), None, Some(7), None]);
        let b = Int32Array::from(vec![None, None, Some(6), Some(7)]);
        let c = add(&a, &b).unwrap();
        assert_eq!(true, c.is_null(0));
        assert_eq!(true, c.is_null(1));
        assert_eq!(false, c.is_null(2));
        assert_eq!(true, c.is_null(3));
        assert_eq!(13, c.value(2));
    }
}
