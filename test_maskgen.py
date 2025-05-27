import numpy as np
import paddle
def gen_casual_document_mask(bz, num_head, seqlen, has_end, causal):
    mask_num = 1
    assert causal == True
    assert has_end == False
    rng = np.random.default_rng()
    sample_indices = rng.choice(seqlen, size=(int)(seqlen/10), replace=False)    
    sample_indices.sort()
    m = np.zeros((bz, num_head, seqlen, mask_num))
    m[:,:,:sample_indices[0],:] = sample_indices[0]
    for i in range(sample_indices.shape[0]-1):
        idx0 = sample_indices[i]
        idx1 = sample_indices[i+1]
        m[:, :, idx0:idx1, 0] = idx1
    m[:,:,sample_indices[-1]:,:] = seqlen-1

    return paddle.to_tensor(m, dtype="int32") 

a = gen_casual_document_mask(1, 16, 2048, False, True)