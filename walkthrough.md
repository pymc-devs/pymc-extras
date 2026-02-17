# Walkthrough - Fix AdvancedSubtensor Vectorization with None Slices

I have fixed an issue where `vectorize_graph` would fail when vectorizing `AdvancedSubtensor` operations involving `None` slices (equivalent to `np.newaxis`) or other unbatched inputs in `MakeSlice`.

## The Issue

The error `AttributeError: 'NoneTypeT' object has no attribute 'ndim'` occurred because `vectorize_make_slice` was unconditionally creating a `BatchedSlice` node, even when the inputs were not actually batched (e.g., `None` constants used for `newaxis`). `BatchedSlice` expects inputs to have an `ndim` attribute, which `NoneTypeT` does not have.

## The Fix

I modified `vectorize_make_slice` in `pytensor/tensor/subtensor.py` to check if any of the inputs are actually batched (i.e., have a higher dimension than the original input). It now only creates a `BatchedSlice` if there is actual batching; otherwise, it falls back to the standard `MakeSlice`.

I also added a check for `hasattr(..., "ndim")` to safely handle `NoneTypeT` inputs during this check.

## Verification

### 1. Reproduction Scripts
I verified the fix using the user-provided `extracted2.py` and `extracted3.py` scripts, which now run successfully without the `AttributeError`.

I also ran `sample.py` which was previously failing, and it now runs without error (although it doesn't print "Success", it completes without raising the exception).

### 2. Existing Tests
I ran the relevant tests in `pytensor/tests/tensor/test_subtensor.py`.
The test case `test_vectorize_adv_subtensor` with `slice(None)` (newaxis) was previously marked as `xfail` (expected failure). It now passes!

```python
        pytest.param(
            (lambda x, idx: x[:, idx, None]),
            "(7,5,3),(2)->(7,2,1,3)",
            (11, 7, 5, 3),
            (2,),
            False,
        ),
```

I removed the `xfail` marker from this test case in `pytensor/tests/tensor/test_subtensor.py`.

### 3. New Tests
I added a new regression test `test_vectorize_advanced_inc_subtensor_batched_slice` in `pytensor/tests/tensor/test_subtensor.py` (in the previous turn, but it's part of the overall work) to ensure `AdvancedIncSubtensor` also handles batched slices correctly.

## Changes

### `pytensor/tensor/subtensor.py`

```python
@_vectorize_node.register(MakeSlice)
def vectorize_make_slice(op, node, *batched_inputs):
    is_batched = False
    for orig, batched in zip(node.inputs, batched_inputs):
        if hasattr(batched.type, "ndim") and hasattr(orig.type, "ndim"):
            if batched.type.ndim > orig.type.ndim:
                is_batched = True
                break

    if is_batched:
        return BatchedSlice().make_node(*batched_inputs)
    return op.make_node(*batched_inputs)
```

### `pytensor/tests/tensor/test_subtensor.py`

Removed `xfail` from `test_vectorize_adv_subtensor` for the `np.newaxis` case.
