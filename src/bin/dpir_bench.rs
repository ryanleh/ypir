use log::debug;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde_json::Value;

use spiral_rs::{arith::*};

use ypir::{
    convolution::{Convolution, negacyclic_perm_u32}, client::*, server::*, scheme::SEED_0,
};

use std::time::{Duration, Instant};


fn compute_hint() -> Vec<u64> {
    let n = 4096;
    let db_rows = 8945;
    let db_cols = 8945;
    let db = vec![0u32; db_rows * db_cols];

    let lwe_params = LWEParams::default();
    let conv = Convolution::new(n);


    let mut hint_0 = vec![0u64; n * db_cols];
    let convd_len = conv.params().crt_count * conv.params().poly_len;

    let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
    let mut v_nega_perm_a = Vec::new();
    for _ in 0..db_rows / n {
        let mut a = vec![0u32; n];
        for idx in 0..n {
            a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
        }
        let nega_perm_a = negacyclic_perm_u32(&a);
        let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
        v_nega_perm_a.push(nega_perm_a_ntt);
    }

    // limit on the number of times we can add results modulo M before we wrap
    let log2_conv_output =
        log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
    println!("{:?}", log2_conv_output);
    let log2_modulus = log2(conv.params().modulus);
    let log2_max_adds = log2_modulus - log2_conv_output - 1;
    assert!(log2_max_adds > 0);
    let max_adds = 1 << log2_max_adds;

    for col in 0..db_cols {
        let mut tmp_col = vec![0u64; convd_len];
        for outer_row in 0..db_rows / n {
            let start_idx = col * db_rows + outer_row * n;
            let pt_col = &db[start_idx..start_idx + n];

            let pt_col_u32 = pt_col
                .iter()
                .map(|&x| x.to_u64() as u32)
                .collect::<Vec<_>>();
            assert_eq!(pt_col_u32.len(), n);
            let pt_ntt = conv.ntt(&pt_col_u32);

            let convolved_ntt = conv.pointwise_mul(&v_nega_perm_a[outer_row], &pt_ntt);

            for r in 0..convd_len {
                tmp_col[r] += convolved_ntt[r] as u64;
            }

            if outer_row % max_adds == max_adds - 1 || outer_row == db_rows / n - 1 {
                let mut col_poly_u32 = vec![0u32; convd_len];
                for i in 0..conv.params().crt_count {
                    for j in 0..conv.params().poly_len {
                        let val = barrett_coeff_u64(
                            conv.params(),
                            tmp_col[i * conv.params().poly_len + j],
                            i,
                        );
                        col_poly_u32[i * conv.params().poly_len + j] = val as u32;
                    }
                }
                let col_poly_raw = conv.raw(&col_poly_u32);
                for i in 0..n {
                    hint_0[i * db_cols + col] += col_poly_raw[i] as u64;
                    hint_0[i * db_cols + col] %= 1u64 << 32;
                }
                tmp_col.fill(0);
            }
        }
    }
    hint_0
}

fn main() {
    let mut total = 0.0;
    let num = 10;
    for _ in 0..num {
        let start = Instant::now();
        compute_hint();
        let got = start.elapsed().as_secs_f64();
        println!("TIME: {:.3}s", got);
        total += got;
    }
    println!("Avg time: {:.2}s", total / (num as f64));
}
