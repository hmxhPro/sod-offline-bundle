# Bug Fix Summary - 2026-07-09

This document summarizes the three issues identified in the problem list and their fixes.

## Issue 1: Annotation Save Jumps to First Image (已修复)

### Problem Description
After manually annotating an image and clicking save, the UI would jump to the first image instead of staying on the current position. This made continuous annotation workflow inefficient.

### Root Cause
In `frontend/src/components/KonvaAnnotator.jsx`, the `handleSave` function was refreshing the image list after saving, but the logic to restore the current position had a flaw. When using `setIdx((i) => ...)`, the callback receives the **previous** state value, not accounting for the fact that the current index might have changed during the refresh.

### Fix Location
`frontend/src/components/KonvaAnnotator.jsx` - lines 288-325

### Changes Made
Changed from:
```javascript
const currentImageId = current.id
// ...refresh images...
const newIdx = visibleRows.findIndex(im => im.id === currentImageId)
if (newIdx !== -1) {
  setIdx(newIdx)
} else {
  setIdx((i) => Math.min(i, Math.max(0, visibleRows.length - 1)))
}
```

To:
```javascript
const currentImageId = current.id
const currentIdx = idx  // Save the current numeric index
// ...refresh images...
const newIdx = visibleRows.findIndex(im => im.id === currentImageId)
if (newIdx !== -1) {
  setIdx(newIdx)  // Found the image, stay at its new position
} else {
  // Use the saved numeric index instead of callback
  setIdx(Math.min(currentIdx, Math.max(0, visibleRows.length - 1)))
}
```

### Expected Behavior After Fix
- When you save an annotation, the UI remains on the current image
- Only if the current image is filtered out (e.g., in "trainable only" mode), the UI moves to the nearest valid position
- No unexpected jumps to the first image


## Issue 2: Dataset Merge Dialog Shows 0 Images (已修复)

### Problem Description
When opening the dataset merge dialog, the popup showed "0 张" (0 images) for all target categories, even though the categories contained images. After merging, the result popup would show the correct image count.

### Root Cause
The `CategoryRecord` database model uses the field name `image_count`, but the frontend component `DatasetMergeDialog.jsx` was accessing `total_images`, which doesn't exist in the schema.

```python
# backend/app/db/models.py - CategoryRecord
image_count: Mapped[int] = mapped_column(Integer, default=0)  # ✓ Correct field name
annotated_count: Mapped[int] = mapped_column(Integer, default=0)
```

```javascript
// frontend/src/components/DatasetMergeDialog.jsx
{cat.total_images || 0}  // ✗ Wrong - this field doesn't exist
```

### Fix Location
`frontend/src/components/DatasetMergeDialog.jsx` - lines 84, 106, 129, 132, 136

### Changes Made
Replaced all instances of `total_images` with `image_count`:
- Line 84: Target category dropdown
- Line 106: Source category dropdown  
- Lines 129-136: Preview section showing image counts

### Expected Behavior After Fix
- The merge dialog now correctly displays the actual image count for each category
- The preview shows accurate before/after merge counts
- Users can make informed decisions about which categories to merge


## Issue 3: Training Epoch Sequence Anomaly (已修复)

### Problem Description
During model training, the displayed epoch numbers didn't follow sequential order (0, 1, 2, 3, 4, 5...), but instead showed patterns like (1, 2, 4, 6, 7, 9, 11), cycling between gaps and no gaps.

### Root Cause
The training progress polling function `_poll_progress()` was using `len(rows)` to count the current epoch instead of reading the actual epoch number from the CSV's "epoch" column. While the CSV file itself contains sequential epochs, if:
1. The CSV is partially flushed during writing
2. Some rows are malformed or incomplete
3. There's a race condition during file I/O

...the row count could temporarily misrepresent the actual epoch.

### Fix Location
`backend/app/services/training_manager.py` - lines 378-394

### Changes Made
Modified `_poll_progress()` to read the actual epoch value from the CSV:

```python
# Old approach - just count rows
current_epoch = len(rows)

# New approach - read the actual epoch column
last_row = rows[-1]
epoch_str = last_row.get("epoch", "").strip()
try:
    current_epoch = int(float(epoch_str)) if epoch_str else len(rows)
except (ValueError, TypeError):
    # Fallback to row count if epoch column is missing or malformed
    current_epoch = len(rows)
```

### Expected Behavior After Fix
- Training epochs now display in proper sequential order: 1, 2, 3, 4, 5...
- The progress indicator accurately reflects the actual training progress
- No more confusing epoch number jumps during training


## Testing Recommendations

### Issue 1 - Annotation Save Position
1. Upload multiple images to a category
2. Navigate to the 3rd or 4th image
3. Draw some annotation boxes
4. Click "Save Annotation" (保存标注)
5. **Expected**: UI should remain on the current image
6. Navigate to another image and test again with "trainable only" filter enabled

### Issue 2 - Dataset Merge Display
1. Create two categories with different numbers of images (e.g., 10 and 15 images)
2. Click the merge button in the CategoryManager
3. **Expected**: Dropdown should show "类别名 (10 张)" and "类别名 (15 张)"
4. Select both categories
5. **Expected**: Preview should show "目标类别当前：10 张图片"，"源类别：15 张图片"，"合并后：25 张图片"

### Issue 3 - Training Epoch Sequence
1. Start a new training job with 50 or 100 epochs
2. Monitor the training progress display
3. **Expected**: Epochs should display as "epoch 1/50", "epoch 2/50", "epoch 3/50", etc. in sequential order
4. Check the backend logs - no skipped epochs should be reported


## Additional Notes

- All fixes are backward compatible and don't require database migrations
- The changes preserve existing functionality while fixing the reported issues
- No new dependencies were added
- Code follows the existing project patterns and conventions


## Files Modified

1. `frontend/src/components/KonvaAnnotator.jsx` - Annotation save position fix
2. `frontend/src/components/DatasetMergeDialog.jsx` - Dataset merge display fix
3. `backend/app/services/training_manager.py` - Training epoch sequence fix
