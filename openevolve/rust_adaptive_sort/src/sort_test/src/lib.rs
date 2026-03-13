use std::thread;

// EVOLVE-BLOCK-START
//
const PARTITION_PARALLEL_THRESHOLD: usize = 10_000;
const TOTAL_PARALLEL_THRESHOLD: usize = 50_000;
const MAX_RECURSION_DEPTH: usize = 128;

pub fn adaptive_sort<T: Ord + Clone + Send>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    parallel_quicksort_depth(arr, 0);
}

fn parallel_quicksort_depth<T: Ord + Clone + Send>(arr: &mut [T], depth: usize) {
    let len = arr.len();
    if len <= 1 {
        return;
    }

    // Stop parallel recursion for small arrays or too deep recursion
    if len < TOTAL_PARALLEL_THRESHOLD || depth > MAX_RECURSION_DEPTH {
        quicksort(arr, 0, len - 1);
        return;
    }

    let pivot_index = partition(arr, 0, len - 1);
    let (left, right) = arr.split_at_mut(pivot_index);
    let right = &mut right[1..]; // skip pivot

    let left_len = left.len();
    let right_len = right.len();

    // Spawn threads for both sides if partitions are large enough
    if left_len >= PARTITION_PARALLEL_THRESHOLD && right_len >= PARTITION_PARALLEL_THRESHOLD {
        thread::scope(|scope| {
            let left_handle = scope.spawn(|| parallel_quicksort_depth(left, depth + 1));
            let right_handle = scope.spawn(|| parallel_quicksort_depth(right, depth + 1));

            // Wait for both threads
            left_handle.join().unwrap();
            right_handle.join().unwrap();
        });
    } else if left_len >= PARTITION_PARALLEL_THRESHOLD {
        // Spawn only left, sort right in current thread
        thread::scope(|scope| {
            scope.spawn(|| parallel_quicksort_depth(left, depth + 1));
            parallel_quicksort_depth(right, depth + 1);
        });
    } else if right_len >= PARTITION_PARALLEL_THRESHOLD {
        // Spawn only right, sort left in current thread
        thread::scope(|scope| {
            scope.spawn(|| parallel_quicksort_depth(right, depth + 1));
            parallel_quicksort_depth(left, depth + 1);
        });
    } else {
        // Both partitions small → sequential recursion
        parallel_quicksort_depth(left, depth + 1);
        parallel_quicksort_depth(right, depth + 1);
    }
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
    // Median-of-three pivot selection
    let mid = low + (high - low) / 2;
    if arr[mid] < arr[low] {
        arr.swap(low, mid);
    }
    if arr[high] < arr[low] {
        arr.swap(low, high);
    }
    if arr[mid] < arr[high] {
        arr.swap(mid, high);
    }

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

// Helpers (unchanged)
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
