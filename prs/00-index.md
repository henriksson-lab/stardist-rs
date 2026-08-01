# Candle PR Submission Index

Fork working tree: `/home/mahogny/github/claude/candle`

General workflow for each PR:

1. `cd /home/mahogny/github/claude/candle`
2. Check the branch is clean: `git status --short`
3. Push the branch: `git push -u <your-fork-remote> <branch>`
4. Open the PR against the listed base branch.

If using GitHub CLI, replace `<your-fork-remote>` with the fork remote name and
run the `gh pr create` command shown in each file.

Suggested order:

1. `01-cpu-conv2d-1x1-fast-path.md`
2. `02-cpu-conv2d-tiled-direct-strides.md`
3. `03-cpu-conv2d-3x3-same-fast-path.md`
4. `04-conv3d-core-cpu.md`
5. `05-conv3d-cuda.md`
6. `06-conv3d-metal.md`
7. `07-nn-conv3d.md`
8. `08-cuda-sm75-compat.md`
9. `09-cuda-gemm-broadcast-collapse.md`
10. `10-cuda-softmax-1024-f32.md`
11. `11-cuda-last-dim-bias-add.md`
12. `12-sam-image-encoder-batched-pos-embed.md`
13. `13-sam-relative-position-add-cuda.md`

Notes:

- PR 3 is stacked on PR 2.
- PR 5, PR 6, and PR 7 are stacked on PR 4.
- PR 8, PR 9, PR 10, and PR 11 are independent of each other and all open
  against `main`. They were originally stacked on PR 8, but none of PR 9-11
  touch `candle-kernels/src/compatibility.cuh`; that base only made them
  buildable on local sm75 hardware. Each is now rebased directly onto `main`,
  with local retest branches `validation/<branch>-sm75` (PR 8 + the one commit)
  that are not submitted.
- PR 12 is independent of the CUDA performance stack and can be opened against
  `main`.
- PR 13 is stacked on PR 12. It should exclude temporary
  `ImageEncoderViTTiming` instrumentation and include only the CUDA
  relative-position add fast path plus its focused test.
- PR 6 needs macOS/Metal runtime validation before submission.
- PR 5 CUDA tests were validated on `validation/conv3d-cuda-sm75`, which
  combines clean PR 5 with PR 8 for local sm75/CUDA 12.8 testability.
