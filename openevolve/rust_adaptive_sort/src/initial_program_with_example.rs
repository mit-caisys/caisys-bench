use std::thread;

// Thresholds
const CHUNK_PARALLEL_THRESHOLD: usize = 10_000;

// Example parallel program that calculates summation
fn parallel_sum(nums: &[i64]) -> i64 {
    // Base case: small slice → sum directly
    if nums.len() <= CHUNK_PARALLEL_THRESHOLD {
        return nums.iter().map(|&x| x as i64).sum();
    }

    // Split in half and sum each side in parallel
    let mid = nums.len() / 2;
    let (left, right) = nums.split_at(mid);

    thread::scope(|scope| {
        // Spawn a thread for the left half
        let left_handle = scope.spawn(|| parallel_sum(left));

        // Compute right half in current thread
        let right_sum = parallel_sum(right);

        // Wait for left result
        let left_sum = left_handle.join().unwrap();

        left_sum + right_sum
    })
}

// Adaptive Sorting Algorithm Implementation
// This program implements a sorting algorithm that can be evolved to adapt to different data patterns

use std::cmp::Ordering;

// EVOLVE-BLOCK-START
// Initial implementation: Simple quicksort
// This can be evolved to:
// - Hybrid algorithms (introsort, timsort-like)
// - Adaptive pivot selection
// - Special handling for nearly sorted data
// - Switching to different algorithms based on data characteristics

pub fn adaptive_sort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }

    // Use quicksort as the base implementation
    quicksort(arr, 0, arr.len() - 1);
}

fn quicksort<T: Ord + Clone>(arr: &mut [T], mut low: usize, mut high: usize) {
    while low < high {
        let pivot_index = partition(arr, low, high);

        // Calculate size of left and right partitions (relative to low/high indices)
        let left_size = pivot_index - low;
        let right_size = high - pivot_index; // This is the size from pivot+1 to high

        if left_size < right_size {
            // Recurse on the smaller (left) side
            if pivot_index > 0 {
                quicksort(arr, low, pivot_index - 1);
            }
            // Loop on the larger (right) side by setting low for the next iteration
            low = pivot_index + 1;
        } else {
            // Recurse on the smaller (right) side
            if pivot_index < high {
                quicksort(arr, pivot_index + 1, high);
            }
            // Loop on the larger (left) side by setting high for the next iteration
            high = if pivot_index == 0 { 0 } else { pivot_index - 1 };
        }
    }
}

fn partition<T: Ord + Clone>(arr: &mut [T], low: usize, high: usize) -> usize {
    // Choose the last element as pivot (can be evolved to use better strategies)
    let pivot = arr[high].clone();
    let mut i = low;

    for j in low..high {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, high);
    i
}

// Helper function to detect if array is nearly sorted
fn is_nearly_sorted<T: Ord>(arr: &[T], threshold: f64) -> bool {
    if arr.len() <= 1 {
        return true;
    }

    let mut inversions = 0;
    let max_inversions = ((arr.len() * (arr.len() - 1)) / 2) as f64 * threshold;

    for i in 0..arr.len() - 1 {
        for j in i + 1..arr.len() {
            if arr[i] > arr[j] {
                inversions += 1;
                if inversions as f64 > max_inversions {
                    return false;
                }
            }
        }
    }

    true
}

// Helper function for insertion sort (useful for small arrays)
fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}
// EVOLVE-BLOCK-END

// Benchmark function to test the sort implementation
pub fn run_benchmark(test_data: Vec<Vec<i32>>) -> BenchmarkResults {
    let mut results = BenchmarkResults {
        times: Vec::new(),
        correctness: Vec::new(),
        adaptability_score: 0.0,
    };

    for data in test_data {
        let mut arr = data.clone();
        let start = std::time::Instant::now();

        adaptive_sort(&mut arr);

        let elapsed = start.elapsed();
        results.times.push(elapsed.as_secs_f64());

        // Check if correctly sorted
        let is_sorted = arr.windows(2).all(|w| w[0] <= w[1]);
        results.correctness.push(is_sorted);
    }

    // Calculate adaptability score based on performance variance
    if results.times.len() > 1 {
        let mean_time: f64 = results.times.iter().sum::<f64>() / results.times.len() as f64;
        let variance: f64 = results
            .times
            .iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>()
            / results.times.len() as f64;

        // Lower variance means better adaptability
        results.adaptability_score = 1.0 / (1.0 + variance.sqrt());
    }

    results
}

#[derive(Debug)]
pub struct BenchmarkResults {
    pub times: Vec<f64>,
    pub correctness: Vec<bool>,
    pub adaptability_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sort() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_empty_array() {
        let mut arr: Vec<i32> = vec![];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![]);
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![42];
        adaptive_sort(&mut arr);
        assert_eq!(arr, vec![42]);
    }
}
