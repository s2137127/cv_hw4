import cv2
import numpy as np
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w,ch = Il.shape
    # labels = np.zeros((h, w), dtype=np.float32)
    Il = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    Il_ = Il.astype(np.float32).copy()
    Ir_ = Ir.astype(np.float32).copy()

    k_size = 5
    k = int(np.floor(k_size / 2))
    Il = np.pad(Il, ((k, k), (k, k)))
    Ir = np.pad(Ir, ((k, k), (k, k)))
    mid = int((k_size ** 2 - 1) / 2)
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    Il_bit = np.zeros((h, w, k_size ** 2), dtype=bool)
    Ir_bit = np.zeros((h, w, k_size ** 2), dtype=bool)
    cost_arr = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_arr_r = np.zeros((h, w, max_disp), dtype=np.float32)
    for i in range(k, h - k):
        for j in range(k, w - k):
            kernel_l = Il[i - k:i + k + 1, j - k:j + k + 1].flatten()
            bit_l = kernel_l < kernel_l[mid]
            Il_bit[i - k, j - k] = bit_l
            kernel_r = Ir[i - k:i + k + 1, j - k:j + k + 1].flatten()
            bit_r = kernel_r < kernel_r[mid]
            Ir_bit[i - k, j - k] = bit_r

    for i in range(h):
        for j in range(w):
            # cost = []
            for d in range(max_disp):
                if j - d >= 0:
                    cost_arr[i, j, d] = np.sum(Il_bit[i, j, :] == Ir_bit[i, j - d, :])
                else:
                    break
            #         cost.append(np.sum(Il_bit[i,j,:] == Ir_bit[i,j-d,:]))
            # labels[i, j ] = np.argmax(cost)
    for i in range(h):
        for j in range(w):
            # cost = []
            for d in range(max_disp):
                if j + d < w:
                    cost_arr_r[i, j, d] = np.sum(Ir_bit[i, j, :] == Il_bit[i, j + d, :])
                else:
                    break
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for i in range(max_disp):
        cost_arr[..., i] = xip.jointBilateralFilter(Il_, cost_arr[..., i], -1, 7, 13)
        cost_arr_r[..., i] = xip.jointBilateralFilter(Ir_, cost_arr_r[..., i], -1, 7, 13)
    # labels = xip.jointBilateralFilter(Il_,labels,-1,7,13)
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    labels = np.argmax(cost_arr, axis=2)
    labels_r = np.argmax(cost_arr_r, axis=2)
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    left_right = np.zeros((h, w))
    holes = []
    # print(labels.shape)
    for i in range(h):
        for j in range(w):
            if labels[i, j] != labels_r[i, j - labels[i, j]]:
                left_right[i, j] = -1
                holes.append((i, j))
    for i, j in holes:
        Fl = Fr = max_disp
        for k in range(1, j):
            if left_right[i, j - k] != -1:
                Fl = labels[i, j - k]
                break
        for k in range(j+1,w):
            if left_right[i, k] != -1:
                Fr = labels[i, k]
                break
        # print(Fl,Fr)
        labels[i,j] = min(Fl,Fr)
    labels = xip.weightedMedianFilter(Il_.astype(np.uint8),labels.astype(np.uint8),7)
    return labels.astype(np.uint8)
