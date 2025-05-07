import torch

def compute_hypervolume(Y, ref_point):
    hv = 0.0
    dom = []
    for i in range(Y.size(0)):
        skip = False
        for j in range(Y.size(0)):
            if j!=i and Y[j,0]>=Y[i,0] and Y[j,1]<=Y[i,1]:
                if Y[j,0]>Y[i,0] or Y[j,1]<Y[i,1]:
                    skip = True
                    break
        if not skip:
            dom.append(Y[i])
    for p in dom:
        hv += (p[0]-ref_point[0])*(ref_point[1]-p[1])
    return hv

def expected_hv_improvement(gp_model, Xcand, Y, ref_point):
    with torch.no_grad():
        mean, var = gp_model.predict(Xcand)
        hv_base = compute_hypervolume(Y, ref_point)
        ei = []
        for i in range(Xcand.size(0)):
            m = mean[i]
            candY = torch.cat([Y, m.unsqueeze(0)], dim=0)
            hv_cand = compute_hypervolume(candY, ref_point)
            ei.append(hv_cand - hv_base)
        return torch.tensor(ei)
