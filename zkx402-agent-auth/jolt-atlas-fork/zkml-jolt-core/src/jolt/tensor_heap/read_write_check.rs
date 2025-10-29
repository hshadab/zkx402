use crate::jolt::execution_trace::JoltONNXCycle;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_core::{
    field::{JoltField, OptimizedMul},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::{AppendToTranscript, Transcript},
    },
};
use onnx_tracer::constants::MAX_TENSOR_SIZE;
use rayon::{iter::IntoParallelIterator, prelude::*};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct ReadWriteCheckingProof<F: JoltField, ProofTranscript: Transcript> {
    /// Joint sumcheck proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub ts1_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    pub ts1_rv_claim: F,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub ts2_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    pub ts2_rv_claim: F,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub ts3_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    pub ts3_rv_claim: F,
    /// The claimed evaluation ra(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub gather_ra_claim: F,
    /// The claimed evaluation rv(r') proven by the read-checking sumcheck.
    pub gather_rv_claim: F,
    /// The claimed evaluation wa(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub rd_wa_claim: F,
    /// The claimed evaluation wv(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub rd_wv_claim: F,
    /// The claimed evaluation val(r_address, r_cycle) output by the read/write-
    /// checking sumcheck.
    pub val_claim: F,
    /// The claimed evaluation Inc(r, r') proven by the write-checking sumcheck.
    pub inc_claim: F,
    /// The sumcheck round index at which we switch from binding cycle variables
    /// to binding address variables.
    pub sumcheck_switch_index: usize,
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove")]
    pub fn prove(
        trace: &[JoltONNXCycle],
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        const DEGREE: usize = 3;
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        debug_assert_eq!(trace.len() * MAX_TENSOR_SIZE, T);

        // Used to batch the read-checking and write-checking sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();
        let z_squared = z.square();
        let z_cubed = z_squared * z;
        let z_fourth = z_cubed * z;

        let num_rounds = K.log_2() + T.log_2();
        let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
        let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let td_writes: Vec<(usize, u64, u64)> = trace
            .iter()
            .flat_map(|cycle| {
                let (address, pre_value, post_value) = cycle.td_write();
                (0..MAX_TENSOR_SIZE).map(move |i| (address[i], pre_value[i], post_value[i]))
            })
            .collect();

        // TODO(Forpee): refactor to use witness generator
        let ts1_rv: Vec<u64> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts1_read().1)
            .collect();
        let ts1_rv = MultilinearPolynomial::from(ts1_rv);

        let ts2_rv: Vec<u64> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts2_read().1)
            .collect();
        let ts2_rv = MultilinearPolynomial::from(ts2_rv);

        let ts3_rv: Vec<u64> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts3_read().1)
            .collect();
        let ts3_rv = MultilinearPolynomial::from(ts3_rv);

        let gather_rv: Vec<u64> = trace
            .par_iter()
            .flat_map(|cycle| cycle.gather_read_values())
            .collect();
        let gather_rv = MultilinearPolynomial::from(gather_rv);

        let rd_wv: Vec<u64> = trace
            .par_iter()
            .flat_map(|cycle| cycle.td_write().2)
            .collect();
        let mut rd_wv = MultilinearPolynomial::from(rd_wv);

        let ts1_addr: Vec<usize> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts1_read().0)
            .collect();
        let ts2_addr: Vec<usize> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts2_read().0)
            .collect();
        let ts3_addr: Vec<usize> = trace
            .par_iter()
            .flat_map(|cycle| cycle.ts3_read().0)
            .collect();

        let gather_addr: Vec<usize> = trace
            .par_iter()
            .flat_map(|cycle| cycle.gather_addresses())
            .collect();
        let td_addr: Vec<usize> = trace
            .par_iter()
            .flat_map(|cycle| cycle.td_write().0)
            .collect();

        // commented out because it makes tests slow
        // #[cfg(test)]
        // let mut val_test = {
        //     // Compute Val in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut val: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
        //         let mut current_val = F::zero();
        //         for j in 0..T {
        //             val_k[j] = current_val;
        //             let (address, _, write_value) = td_writes[j];
        //             if address == k {
        //                 current_val = F::from_u64(write_value);
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(val)
        // };
        // #[cfg(test)]
        // let mut ts1_ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if ts1_addr[j] == k {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };
        // #[cfg(test)]
        // let mut ts2_ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if ts2_addr[j] == k {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };
        // #[cfg(test)]
        // let mut ts3_ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if ts3_addr[j] == k {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };
        // #[cfg(test)]
        // let mut gather_ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if gather_addr[j] == k {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };
        // #[cfg(test)]
        // let mut wa_test = {
        //     // Compute wa in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut wa: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     wa.par_chunks_mut(T).enumerate().for_each(|(k, wa_k)| {
        //         for j in 0..T {
        //             if td_addr[j] ==

        //                 wa_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(wa)
        // };

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<Vec<i64>> = td_writes[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .map(|trace_chunk| {
                let mut delta = vec![0i64; K];
                for cycle in trace_chunk.iter() {
                    let (k, pre_value, post_value) = cycle;
                    let increment = *post_value as i64 - *pre_value as i64;
                    debug_assert!(*k != 0 || increment == 0, "{cycle:?}"); // Zero register
                    delta[*k] += increment;
                }
                debug_assert_eq!(delta[0], 0); // Zero register
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i64>> = Vec::with_capacity(num_chunks);
        checkpoints.push(vec![0; K]);

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint: Vec<i64> = checkpoints[chunk_index]
                .par_iter()
                .zip(delta.into_par_iter())
                .map(|(val_k, delta_k)| val_k + delta_k)
                .collect::<Vec<_>>();
            debug_assert_eq!(next_checkpoint[0], 0); // Zero register
            checkpoints.push(next_checkpoint);
        }
        // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
        // Generate checkpoints as a flat vector because it will be turned into the
        // materialized Val polynomial after the first half of sumcheck.
        let mut val_checkpoints: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = F::from_i64(*src))
            });

        drop(_guard);
        drop(span);

        // #[cfg(test)]
        // {
        //     // Check that checkpoints are correct
        //     for (chunk_index, checkpoint) in val_checkpoints.chunks(K).enumerate() {
        //         let j = chunk_index * chunk_size;
        //         for (k, V_k) in checkpoint.iter().enumerate() {
        //             assert_eq!(
        //                 *V_k,
        //                 val_test.get_bound_coeff(k * T + j),
        //                 "k = {k}, j = {j}"
        //             );
        //         }
        //     }
        // }

        // A table that, in round i of sumcheck, stores all evaluations
        //     EQ(x, r_i, ..., r_1)
        // as x ranges over {0, 1}^i.
        // (As described in "Computing other necessary arrays and worst-case
        // accounting", Section 8.2.2)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        let span = tracing::span!(
            tracing::Level::INFO,
            "compute I (increments data structure)"
        );
        let _guard = span.enter();

        // Data structure described in Equation (72)
        let mut I: Vec<Vec<(usize, usize, F, F)>> = td_writes
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                trace_chunk
                    .iter()
                    .map(|cycle| {
                        let (k, pre_value, post_value) = *cycle;
                        let increment = post_value as i64 - pre_value as i64;
                        let inc = if increment == 0 {
                            (j, k, F::zero(), F::zero())
                        } else {
                            (j, k, F::zero(), F::from_i64(increment))
                        };
                        j += 1;
                        inc
                    })
                    .collect()
            })
            .collect();

        drop(_guard);
        drop(span);

        // rv(r')
        let (rv_evals, eq_r_prime) = MultilinearPolynomial::batch_evaluate(
            &[&ts1_rv, &ts2_rv, &gather_rv, &ts3_rv],
            &r_prime,
        );
        let (ts1_rv_eval, ts2_rv_eval, gather_rv_eval, ts3_rv_eval) =
            (rv_evals[0], rv_evals[1], rv_evals[2], rv_evals[3]);
        // eq(r, k)
        let mut eq_r = MultilinearPolynomial::from(EqPolynomial::evals(&r));
        // eq(r', j)
        let mut eq_r_prime = MultilinearPolynomial::from(eq_r_prime);

        let span = tracing::span!(tracing::Level::INFO, "compute Inc(r, r')");
        let _guard = span.enter();

        // Inc(r, r')
        let inc_eval: F = td_writes
            .par_iter()
            .enumerate()
            .map(|(j, cycle)| {
                let (k, pre_value, post_value) = *cycle;
                let increment = post_value as i64 - pre_value as i64;
                if increment == 0 {
                    F::zero()
                } else {
                    eq_r.get_coeff(k) * eq_r_prime.get_coeff(j) * F::from_i64(increment)
                }
            })
            .sum();

        drop(_guard);
        drop(span);

        // Linear combination of the read-checking claim (which is rv(r')) and the
        // write-checking claim (which is Inc(r, r'))
        let mut previous_claim = inc_eval
            + z * ts1_rv_eval
            + z_squared * ts2_rv_eval
            + z_cubed * gather_rv_eval
            + z_fourth * ts3_rv_eval;
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let span = tracing::span!(
            tracing::Level::INFO,
            "First log(T / num_chunks) rounds of sumcheck"
        );
        let _guard = span.enter();

        /// A collection of vectors that are used in each of the first log(T / num_chunks)
        /// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
        /// across all log(T / num_chunks) rounds.
        struct DataBuffers<F: JoltField> {
            /// Contains
            ///     Val(k, j', 0, ..., 0)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
            val_j_0: Vec<F>,
            /// `val_j_r[0]` contains
            ///     Val(k, j'', 0, r_i, ..., r_1)
            /// `val_j_r[1]` contains
            ///     Val(k, j'', 1, r_i, ..., r_1)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
            val_j_r: [Vec<F>; 2],
            /// `ra[0]` contains
            ///     ra(k, j'', 0, r_i, ..., r_1)
            /// `ra[1]` contains
            ///     ra(k, j'', 1, r_i, ..., r_1)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
            ts1_ra: [Vec<F>; 2],
            ts2_ra: [Vec<F>; 2],
            gather_ra: [Vec<F>; 2],
            ts3_ra: [Vec<F>; 2],
            /// `wa[0]` contains
            ///     wa(k, j'', 0, r_i, ..., r_1)
            /// `wa[1]` contains
            ///     wa(k, j'', 1, r_i, ..., r_1)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
            /// where j'' are the higher (log(T) - i - 1) bits of j'
            rd_wa: [Vec<F>; 2],
            dirty_indices: Vec<usize>,
        }
        let mut data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: vec![F::zero(); K],
                val_j_r: [vec![F::zero(); K], vec![F::zero(); K]],
                ts1_ra: [vec![F::zero(); K], vec![F::zero(); K]],
                ts2_ra: [vec![F::zero(); K], vec![F::zero(); K]],
                gather_ra: [vec![F::zero(); K], vec![F::zero(); K]],
                ts3_ra: [vec![F::zero(); K], vec![F::zero(); K]],
                rd_wa: [vec![F::zero(); K], vec![F::zero(); K]],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        // First log(T / num_chunks) rounds of sumcheck
        for round in 0..chunk_size.log_2() {
            // #[cfg(test)]
            // {
            //     let mut expected_claim = F::zero();
            //     for j in 0..(T >> round) {
            //         let mut inner_sum = F::zero();
            //         for k in 0..K {
            //             let kj = k * (T >> round) + j;
            //             // write-checking sumcheck
            //             inner_sum += eq_r.get_bound_coeff(k)
            //                 * wa_test.get_bound_coeff(kj)
            //                 * (rd_wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj));
            //             // read-checking sumcheck
            //             inner_sum +=
            //                 z * ts1_ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
            //             inner_sum += z_squared
            //                 * ts2_ra_test.get_bound_coeff(kj)
            //                 * val_test.get_bound_coeff(kj);
            //             inner_sum += z_cubed
            //                 * gather_ra_test.get_bound_coeff(kj)
            //                 * val_test.get_bound_coeff(kj);
            //             inner_sum += z_fourth
            //                 * ts3_ra_test.get_bound_coeff(kj)
            //                 * val_test.get_bound_coeff(kj);
            //         }
            //         expected_claim += eq_r_prime.get_bound_coeff(j) * inner_sum;
            //     }
            //     assert_eq!(
            //         expected_claim, previous_claim,
            //         "Sumcheck sanity check failed in round {round}"
            //     );
            // }

            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = I
                .par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero(), F::zero()];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ts1_ra,
                        ts2_ra,
                        ts3_ra,
                        gather_ra,
                        rd_wa,
                        dirty_indices,
                    } = buffers;

                    val_j_0.as_mut_slice().copy_from_slice(checkpoint);

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                // ts1
                                let k = ts1_addr[j];

                                dirty_indices.push(k);

                                ts1_ra[0][k] += A[j_bound];

                                // ts2
                                let k = ts2_addr[j];

                                dirty_indices.push(k);

                                ts2_ra[0][k] += A[j_bound];

                                // gather
                                let k = gather_addr[j];

                                dirty_indices.push(k);

                                gather_ra[0][k] += A[j_bound];

                                // ts3
                                let k = ts3_addr[j];

                                dirty_indices.push(k);

                                ts3_ra[0][k] += A[j_bound];

                                // td
                                let k = td_addr[j];

                                dirty_indices.push(k);

                                rd_wa[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);

                                // ts1
                                let k = ts1_addr[j];

                                dirty_indices.push(k);

                                ts1_ra[1][k] += A[j_bound];

                                // ts2
                                let k = ts2_addr[j];

                                dirty_indices.push(k);

                                ts2_ra[1][k] += A[j_bound];

                                // gather
                                let k = gather_addr[j];

                                dirty_indices.push(k);

                                gather_ra[1][k] += A[j_bound];

                                // ts3
                                let k = ts3_addr[j];

                                dirty_indices.push(k);

                                ts3_ra[1][k] += A[j_bound];

                                // td
                                let k = td_addr[j];

                                dirty_indices.push(k);

                                rd_wa[1][k] += A[j_bound];
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }

                            let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                                j_prime / 2,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let wv_evals =
                                rd_wv.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                            let mut inner_sum_evals = [F::zero(); 3];
                            for k in dirty_indices.drain(..) {
                                let mut m_val: Option<F> = None;
                                let mut val_eval_2: Option<F> = None;
                                let mut val_eval_3: Option<F> = None;

                                // rs1 read-checking sumcheck
                                if !ts1_ra[0][k].is_zero() || !ts1_ra[1][k].is_zero() {
                                    // Preemptively multiply by `z` to save a mult
                                    let ra_eval_0 = z * ts1_ra[0][k];
                                    let ra_eval_1 = z * ts1_ra[1][k];
                                    let m_ra = ra_eval_1 - ra_eval_0;
                                    let ra_eval_2 = ra_eval_1 + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    m_val = Some(val_j_r[1][k] - val_j_r[0][k]);
                                    val_eval_2 = Some(val_j_r[1][k] + m_val.unwrap());
                                    val_eval_3 = Some(val_eval_2.unwrap() + m_val.unwrap());

                                    inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                    inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                    inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                    ts1_ra[0][k] = F::zero();
                                    ts1_ra[1][k] = F::zero();
                                }

                                // rs2 read-checking sumcheck
                                if !ts2_ra[0][k].is_zero() || !ts2_ra[1][k].is_zero() {
                                    // Preemptively multiply by `z_squared` to save a mult
                                    let ra_eval_0 = z_squared * ts2_ra[0][k];
                                    let ra_eval_1 = z_squared * ts2_ra[1][k];
                                    let m_ra = ra_eval_1 - ra_eval_0;
                                    let ra_eval_2 = ra_eval_1 + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    m_val = m_val.or(Some(val_j_r[1][k] - val_j_r[0][k]));
                                    val_eval_2 =
                                        val_eval_2.or(Some(val_j_r[1][k] + m_val.unwrap()));
                                    val_eval_3 =
                                        val_eval_3.or(Some(val_eval_2.unwrap() + m_val.unwrap()));

                                    inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                    inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                    inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                    ts2_ra[0][k] = F::zero();
                                    ts2_ra[1][k] = F::zero();
                                }

                                // gather read-checking sumcheck
                                if !gather_ra[0][k].is_zero() || !gather_ra[1][k].is_zero() {
                                    // Preemptively multiply by `z_cubed` to save a mult
                                    let ra_eval_0 = z_cubed * gather_ra[0][k];
                                    let ra_eval_1 = z_cubed * gather_ra[1][k];
                                    let m_ra = ra_eval_1 - ra_eval_0;
                                    let ra_eval_2 = ra_eval_1 + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    m_val = m_val.or(Some(val_j_r[1][k] - val_j_r[0][k]));
                                    val_eval_2 =
                                        val_eval_2.or(Some(val_j_r[1][k] + m_val.unwrap()));
                                    val_eval_3 =
                                        val_eval_3.or(Some(val_eval_2.unwrap() + m_val.unwrap()));

                                    inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                    inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                    inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                    gather_ra[0][k] = F::zero();
                                    gather_ra[1][k] = F::zero();
                                }

                                // ts3 read-checking sumcheck
                                if !ts3_ra[0][k].is_zero() || !ts3_ra[1][k].is_zero() {
                                    // Preemptively multiply by `z_fourth` to save a mult
                                    let ra_eval_0 = z_fourth * ts3_ra[0][k];
                                    let ra_eval_1 = z_fourth * ts3_ra[1][k];
                                    let m_ra = ra_eval_1 - ra_eval_0;
                                    let ra_eval_2 = ra_eval_1 + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    m_val = m_val.or(Some(val_j_r[1][k] - val_j_r[0][k]));
                                    val_eval_2 =
                                        val_eval_2.or(Some(val_j_r[1][k] + m_val.unwrap()));
                                    val_eval_3 =
                                        val_eval_3.or(Some(val_eval_2.unwrap() + m_val.unwrap()));

                                    inner_sum_evals[0] += ra_eval_0.mul_0_optimized(val_j_r[0][k]);
                                    inner_sum_evals[1] += ra_eval_2 * val_eval_2.unwrap();
                                    inner_sum_evals[2] += ra_eval_3 * val_eval_3.unwrap();

                                    ts3_ra[0][k] = F::zero();
                                    ts3_ra[1][k] = F::zero();
                                }

                                // Write-checking sumcheck
                                if !rd_wa[0][k].is_zero() || !rd_wa[1][k].is_zero() {
                                    // Save a mult by multiplying by `eq_r_eval` sooner rather than later
                                    let eq_r_eval = eq_r.get_coeff(k);
                                    let wa_eval_0 = if rd_wa[0][k].is_zero() {
                                        F::zero()
                                    } else {
                                        let eval = eq_r_eval * rd_wa[0][k];
                                        inner_sum_evals[0] += eval * (wv_evals[0] - val_j_r[0][k]);
                                        eval
                                    };
                                    let wa_eval_1 = eq_r_eval.mul_0_optimized(rd_wa[1][k]);
                                    let m_wa = wa_eval_1 - wa_eval_0;
                                    let wa_eval_2 = wa_eval_1 + m_wa;
                                    let wa_eval_3 = wa_eval_2 + m_wa;

                                    let m_val = m_val.unwrap_or(val_j_r[1][k] - val_j_r[0][k]);
                                    let val_eval_2 = val_eval_2.unwrap_or(val_j_r[1][k] + m_val);
                                    let val_eval_3 = val_eval_3.unwrap_or(val_eval_2 + m_val);

                                    inner_sum_evals[1] += wa_eval_2 * (wv_evals[1] - val_eval_2);
                                    inner_sum_evals[2] += wa_eval_3 * (wv_evals[2] - val_eval_3);

                                    rd_wa[0][k] = F::zero();
                                    rd_wa[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }
                            dirty_indices.clear();

                            evals[0] += eq_r_prime_evals[0] * inner_sum_evals[0];
                            evals[1] += eq_r_prime_evals[1] * inner_sum_evals[1];
                            evals[2] += eq_r_prime_evals[2] * inner_sum_evals[2];
                        });

                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r_cycle.insert(0, r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
            let _inner_guard = inner_span.enter();

            // Bind I
            I.par_iter_mut().for_each(|I_chunk| {
                // Note: A given row in an I_chunk may not be ordered by k after binding
                let mut next_bound_index = 0;
                let mut bound_indices: Vec<Option<usize>> = vec![None; K];

                for i in 0..I_chunk.len() {
                    let (j_prime, k, inc_lt, inc) = I_chunk[i];
                    if let Some(bound_index) = bound_indices[k] {
                        if I_chunk[bound_index].0 == j_prime / 2 {
                            // Neighbor was already processed
                            debug_assert!(j_prime % 2 == 1);
                            I_chunk[bound_index].2 += r_j * inc_lt;
                            I_chunk[bound_index].3 += inc;
                            continue;
                        }
                    }
                    // First time this k has been encountered
                    let bound_value = if j_prime % 2 == 0 {
                        // (1 - r_j) * inc_lt + r_j * inc
                        inc_lt + r_j * (inc - inc_lt)
                    } else {
                        r_j * inc_lt
                    };
                    I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                    bound_indices[k] = Some(next_bound_index);
                    next_bound_index += 1;
                }
                I_chunk.truncate(next_bound_index);
            });

            drop(_inner_guard);
            drop(inner_span);

            rayon::join(
                || rd_wv.bind_parallel(r_j, BindingOrder::LowToHigh),
                || eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh),
            );

            // #[cfg(test)]
            // {
            //     val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     ts1_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     ts2_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     gather_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     ts3_ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     wa_test.bind_parallel(r_j, BindingOrder::LowToHigh);

            //     // Check that row indices of I are non-decreasing
            //     let mut current_row = 0;
            //     for I_chunk in I.iter() {
            //         for (row, _, _, _) in I_chunk {
            //             if *row != current_row {
            //                 assert_eq!(*row, current_row + 1);
            //                 current_row = *row;
            //             }
            //         }
            //     }
            // }

            let inner_span = tracing::span!(tracing::Level::INFO, "Update A");
            let _inner_guard = inner_span.enter();

            // Update A for this round (see Equation 55)
            let (A_left, A_right) = A.split_at_mut(1 << round);
            A_left
                .par_iter_mut()
                .zip(A_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });
        }

        drop(_guard);
        drop(span);

        // At this point I has been bound to a point where each chunk contains a single row,
        // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
        // standard sumcheck directly using those polynomials.

        let span = tracing::span!(tracing::Level::INFO, "Materialize ts1_ra polynomial");
        let _guard = span.enter();
        let mut ts1_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        ts1_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, &k) in ts1_addr
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut ts1_ra = MultilinearPolynomial::from(ts1_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize ts2_ra polynomial");
        let _guard = span.enter();
        let mut ts2_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        ts2_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, &k) in ts2_addr
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut ts2_ra = MultilinearPolynomial::from(ts2_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize gather_ra polynomial");
        let _guard = span.enter();
        let mut gather_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        gather_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, &k) in gather_addr
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut gather_ra = MultilinearPolynomial::from(gather_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize ts3_ra polynomial");
        let _guard = span.enter();
        let mut ts3_ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        ts3_ra
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, &k) in ts3_addr
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut ts3_ra = MultilinearPolynomial::from(ts3_ra);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize rd_wa polynomial");
        let _guard = span.enter();
        let mut rd_wa: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        rd_wa
            .par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, wa_chunk)| {
                for (j_bound, &k) in td_addr
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    wa_chunk[k] += A[j_bound];
                }
            });
        let mut rd_wa = MultilinearPolynomial::from(rd_wa);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
        let _guard = span.enter();
        let mut val: Vec<F> = val_checkpoints;
        val.par_chunks_mut(K)
            .zip(I.into_par_iter())
            .enumerate()
            .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                for (j, k, inc_lt, _inc) in I_chunk.into_iter() {
                    debug_assert_eq!(j, chunk_index);
                    val_chunk[k] += inc_lt;
                }
            });
        let mut val = MultilinearPolynomial::from(val);
        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Remaining rounds of sumcheck");
        let _guard = span.enter();

        // Remaining rounds of sumcheck
        for round in 0..num_rounds - chunk_size.log_2() {
            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = if eq_r_prime.len() > 1 {
                // Not done binding cycle variables yet
                (0..eq_r_prime.len() / 2)
                    .into_par_iter()
                    .map(|j| {
                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                        let wv_evals = rd_wv.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);

                        let inner_sum_evals: [F; 3] = (0..K)
                            .into_par_iter()
                            .map(|k| {
                                let index = j * K + k;
                                let ts1_ra_evals =
                                    ts1_ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let ts2_ra_evals =
                                    ts2_ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let gather_ra_evals = gather_ra.sumcheck_evals(
                                    index,
                                    DEGREE,
                                    BindingOrder::HighToLow,
                                );
                                let ts3_ra_evals =
                                    ts3_ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let wa_evals =
                                    rd_wa.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let val_evals =
                                    val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);

                                // Save a mult by multiplying by `eq_r_eval` sooner rather than later
                                let eq_r_eval = eq_r.get_coeff(k);

                                [
                                    eq_r_eval
                                        .mul_0_optimized(wa_evals[0])
                                        .mul_0_optimized(wv_evals[0] - val_evals[0])
                                        + z * ts1_ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_squared * ts2_ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_cubed
                                            * gather_ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_fourth * ts3_ra_evals[0].mul_0_optimized(val_evals[0]),
                                    eq_r_eval
                                        .mul_0_optimized(wa_evals[1])
                                        .mul_0_optimized(wv_evals[1] - val_evals[1])
                                        + z * ts1_ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_squared * ts2_ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_cubed
                                            * gather_ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_fourth * ts3_ra_evals[1].mul_0_optimized(val_evals[1]),
                                    eq_r_eval
                                        .mul_0_optimized(wa_evals[2])
                                        .mul_0_optimized(wv_evals[2] - val_evals[2])
                                        + z * ts1_ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_squared * ts2_ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_cubed
                                            * gather_ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_fourth * ts3_ra_evals[2].mul_0_optimized(val_evals[2]),
                                ]
                            })
                            .reduce(
                                || [F::zero(); 3],
                                |running, new| {
                                    [
                                        running[0] + new[0],
                                        running[1] + new[1],
                                        running[2] + new[2],
                                    ]
                                },
                            );

                        [
                            eq_r_prime_evals[0] * inner_sum_evals[0],
                            eq_r_prime_evals[1] * inner_sum_evals[1],
                            eq_r_prime_evals[2] * inner_sum_evals[2],
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    )
            } else {
                // Cycle variables are fully bound, so:
                // eq(r', r_cycle) is a constant
                let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
                // ...and wv(r_cycle) is a constant
                let wv_eval = rd_wv.final_sumcheck_claim();

                let evals = (0..ts1_ra.len() / 2)
                    .into_par_iter()
                    .map(|k| {
                        let eq_r_evals = eq_r.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let ts1_ra_evals =
                            ts1_ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let ts2_ra_evals =
                            ts2_ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let gather_ra_evals =
                            gather_ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let ts3_ra_evals =
                            ts3_ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let wa_evals = rd_wa.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                        [
                            eq_r_evals[0] * wa_evals[0] * (wv_eval - val_evals[0])
                                + z * ts1_ra_evals[0] * val_evals[0]
                                + z_squared * ts2_ra_evals[0] * val_evals[0]
                                + z_cubed * gather_ra_evals[0] * val_evals[0]
                                + z_fourth * ts3_ra_evals[0] * val_evals[0],
                            eq_r_evals[1] * wa_evals[1] * (wv_eval - val_evals[1])
                                + z * ts1_ra_evals[1] * val_evals[1]
                                + z_squared * ts2_ra_evals[1] * val_evals[1]
                                + z_cubed * gather_ra_evals[1] * val_evals[1]
                                + z_fourth * ts3_ra_evals[1] * val_evals[1],
                            eq_r_evals[2] * wa_evals[2] * (wv_eval - val_evals[2])
                                + z * ts1_ra_evals[2] * val_evals[2]
                                + z_squared * ts2_ra_evals[2] * val_evals[2]
                                + z_cubed * gather_ra_evals[2] * val_evals[2]
                                + z_fourth * ts3_ra_evals[2] * val_evals[2],
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );
                [
                    eq_r_prime_eval * evals[0],
                    eq_r_prime_eval * evals[1],
                    eq_r_prime_eval * evals[2],
                ]
            };

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            previous_claim = univariate_poly.evaluate(&r_j);

            // Bind polynomials
            if eq_r_prime.len() > 1 {
                // Bind a cycle variable j
                r_cycle.insert(round, r_j);
                // Note that `eq_r` is a polynomial over only the address variables,
                // so it is not bound here
                [
                    &mut ts1_ra,
                    &mut ts2_ra,
                    &mut gather_ra,
                    &mut ts3_ra,
                    &mut rd_wa,
                    &mut rd_wv,
                    &mut val,
                    &mut eq_r_prime,
                ]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            } else {
                // Bind an address variable k
                r_address.push(r_j);
                // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
                // variables, so they are not bound here
                [
                    &mut ts1_ra,
                    &mut ts2_ra,
                    &mut gather_ra,
                    &mut ts3_ra,
                    &mut rd_wa,
                    &mut val,
                    &mut eq_r,
                ]
                .into_par_iter()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            }
        }

        let rw_proof = ReadWriteCheckingProof {
            sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
            ts1_ra_claim: ts1_ra.final_sumcheck_claim(),
            ts1_rv_claim: ts1_rv_eval,
            ts2_ra_claim: ts2_ra.final_sumcheck_claim(),
            ts2_rv_claim: ts2_rv_eval,
            ts3_ra_claim: ts3_ra.final_sumcheck_claim(),
            ts3_rv_claim: ts3_rv_eval,
            gather_ra_claim: gather_ra.final_sumcheck_claim(),
            gather_rv_claim: gather_rv_eval,
            rd_wa_claim: rd_wa.final_sumcheck_claim(),
            rd_wv_claim: rd_wv.final_sumcheck_claim(),
            val_claim: val.final_sumcheck_claim(),
            inc_claim: inc_eval,
            sumcheck_switch_index: chunk_size.log_2(),
        };

        drop_in_background_thread((ts1_ra, rd_wv, val, data_buffers, eq_r, eq_r_prime, A));

        (rw_proof, r_address, r_cycle)
    }

    pub fn verify(
        &self,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Vec<F>, Vec<F>) {
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        let z: F = transcript.challenge_scalar();
        let z_squared = z.square();
        let z_cubed = z_squared * z;
        let z_fourth = z_cubed * z;
        let (sumcheck_claim, r_sumcheck) = self
            .sumcheck_proof
            .verify(
                self.inc_claim
                    + z * self.ts1_rv_claim
                    + z.square() * self.ts2_rv_claim
                    + z_cubed * self.gather_rv_claim
                    + z_fourth * self.ts3_rv_claim,
                T.log_2() + K.log_2(),
                3,
                transcript,
            )
            .unwrap();

        // The high-order cycle variables are bound after the switch
        let mut r_cycle = r_sumcheck[self.sumcheck_switch_index..T.log_2()].to_vec();
        // First `sumcheck_switch_index` rounds bind cycle variables from low to high
        r_cycle.extend(r_sumcheck[..self.sumcheck_switch_index].iter().rev());
        // Final log(K) rounds bind address variables
        let r_address = r_sumcheck[T.log_2()..].to_vec();

        // eq(r', r_cycle)
        let eq_eval_cycle = EqPolynomial::mle(&r_prime, &r_cycle);
        // eq(r, r_address)
        let eq_eval_address = EqPolynomial::mle(&r, &r_address);

        let write_term = eq_eval_address
            * eq_eval_cycle
            * self.rd_wa_claim
            * (self.rd_wv_claim - self.val_claim);
        let ts1_term = z * eq_eval_cycle * self.ts1_ra_claim * self.val_claim;
        let ts2_term = z.square() * eq_eval_cycle * self.ts2_ra_claim * self.val_claim;
        let gather_term = z_cubed * eq_eval_cycle * self.gather_ra_claim * self.val_claim;
        let ts3_term = z_fourth * eq_eval_cycle * self.ts3_ra_claim * self.val_claim;
        let lhs = write_term + ts1_term + ts2_term + gather_term + ts3_term;

        if lhs != sumcheck_claim && std::env::var("JOLT_DEBUG_RW").is_ok() {
            eprintln!("[RW-DEBUG] Sumcheck mismatch");
            eprintln!("[RW-DEBUG] z                = {:?}", z);
            eprintln!("[RW-DEBUG] eq_addr         = {:?}", eq_eval_address);
            eprintln!("[RW-DEBUG] eq_cycle        = {:?}", eq_eval_cycle);
            eprintln!("[RW-DEBUG] rd_wa_claim     = {:?}", self.rd_wa_claim);
            eprintln!("[RW-DEBUG] rd_wv_claim     = {:?}", self.rd_wv_claim);
            eprintln!("[RW-DEBUG] val_claim       = {:?}", self.val_claim);
            eprintln!("[RW-DEBUG] ts1_ra_claim    = {:?}", self.ts1_ra_claim);
            eprintln!("[RW-DEBUG] ts1_rv_claim    = {:?}", self.ts1_rv_claim);
            eprintln!("[RW-DEBUG] ts2_ra_claim    = {:?}", self.ts2_ra_claim);
            eprintln!("[RW-DEBUG] ts2_rv_claim    = {:?}", self.ts2_rv_claim);
            eprintln!("[RW-DEBUG] ts3_ra_claim    = {:?}", self.ts3_ra_claim);
            eprintln!("[RW-DEBUG] ts3_rv_claim    = {:?}", self.ts3_rv_claim);
            eprintln!("[RW-DEBUG] gather_ra_claim = {:?}", self.gather_ra_claim);
            eprintln!("[RW-DEBUG] gather_rv_claim = {:?}", self.gather_rv_claim);
            eprintln!("[RW-DEBUG] write_term      = {:?}", write_term);
            eprintln!("[RW-DEBUG] ts1_term        = {:?}", ts1_term);
            eprintln!("[RW-DEBUG] ts2_term        = {:?}", ts2_term);
            eprintln!("[RW-DEBUG] gather_term     = {:?}", gather_term);
            eprintln!("[RW-DEBUG] ts3_term        = {:?}", ts3_term);
            eprintln!("[RW-DEBUG] lhs             = {:?}", lhs);
            eprintln!("[RW-DEBUG] rhs (claim)     = {:?}", sumcheck_claim);
        }

        assert_eq!(lhs, sumcheck_claim, "Read/write-checking sumcheck failed");

        (r_address, r_cycle)
    }
}
