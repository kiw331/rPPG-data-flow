# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Euclidean distance transform (EDT).

On Linux/GPU: uses a Triton kernel for fast batch processing.
On Windows (or when triton is unavailable): falls back to cv2.distanceTransform on CPU.
"""

import torch

# triton is not available on Windows; provide a CPU fallback using cv2
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def edt_kernel(
        inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr
    ):
        """Triton JIT kernel for 1-D parabola-envelope EDT (O(N^2) algorithm)."""
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride),
                    mask=(k + 1) < length,
                    other=q,
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        """
        Compute the Euclidean Distance Transform for a batch of binary images.

        Args:
            data: (B, H, W) bool/uint8 CUDA tensor
        Returns:
            EDT tensor of the same shape.
        """
        assert data.dim() == 3
        assert data.is_cuda
        B, H, W = data.shape
        data = data.contiguous()

        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()

        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        edt_kernel[(B, H)](
            output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=True
        )

        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        edt_kernel[(B, W)](
            output.clone(), output, parabola_loc, parabola_inter, H, W, horizontal=False
        )
        return output.sqrt()

else:
    # ── Windows / CPU fallback ────────────────────────────────────────────────
    import cv2
    import numpy as np

    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        """
        CPU fallback using cv2.distanceTransform (equivalent result, no triton needed).

        Args:
            data: (B, H, W) bool/uint8 tensor (CPU or CUDA)
        Returns:
            EDT tensor of the same shape (on the same device).
        """
        assert data.dim() == 3
        B, H, W = data.shape
        device = data.device
        data_np = data.cpu().numpy().astype(np.uint8)
        output = np.zeros((B, H, W), dtype=np.float32)
        for b in range(B):
            output[b] = cv2.distanceTransform(data_np[b], cv2.DIST_L2, 0)
        return torch.from_numpy(output).to(device)
