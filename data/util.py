import numpy as np
import os,json

RHD2Bighand_skeidx = [0, 4, 8, 12, 16, 20, 3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, 19, 18, 17]
mano2Bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]
STB2Bighand_skeidx = [0, 17, 13, 9, 5, 1, 18, 19, 20, 14, 15, 16, 10, 11, 12, 6, 7, 8, 2, 3, 4]
FreiHand2RHD_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
Bighand2RHD_skeidx = [0, 8, 7, 6, 1, 11, 10, 9, 2, 14, 13, 12, 3, 17, 16, 15, 4, 20, 19, 18, 5]
RHD2FreiHand_skeidx = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
Bighand2mano_skeidx = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]
Bighand2DOED_skeidx = [8,11,14,17,20]
Bighand2DOED_former_skeidx = [7,10,13,16,19]

# For Bighand idx
mmcp_idx = 3
root_idx = mmcp_idx
wrist_idx = 0
mpip_idx = 12
norm_idx = mpip_idx  # mpip
# normlize norm_idx and mmcp to 1

stb_scale = 0.04 # bone length between mmcp and mpip is 4 cm

#From DO dataset
female_bonelength = np.array([8.0,3.8,2.2,2.2,7.5,4.2,2.4,2.2,6.9,2.6,1.6,2.0,7.3,3.4,2.3,2.0,2.1,4.2,3.3,2.7])*0.01 #m #Mano index #from DO
male_bonelength =   np.array([8.8,3.5,2.7,2.5,8.4,4.0,2.7,2.8,7.0,3.5,2.1,2.3,8.0,3.8,2.6,2.6,2.1,4.2,3.5,3.5])*0.01 #m #Mano index #from DO

hand_template = np.array([[-0.0000,  0.0000,  0.0000],  #right hand + MANO index + relative pose from STB
                        [ 0.0248, -0.0744,  0.0066],
                        [ 0.0178, -0.1067, -0.0014],
                        [ 0.0123, -0.1257, -0.0110],
                        [ 0.0069, -0.0768,  0.0124],
                        [-0.0043, -0.0839, -0.0254],
                        [-0.0047, -0.0645, -0.0412],
                        [-0.0247, -0.0586,  0.0184],
                        [-0.0343, -0.0653, -0.0114],
                        [-0.0388, -0.0683, -0.0255],
                        [-0.0092, -0.0736,  0.0166],
                        [-0.0217, -0.0867, -0.0157],
                        [-0.0160, -0.0671, -0.0263],
                        [ 0.0263, -0.0201, -0.0040],
                        [ 0.0282, -0.0327, -0.0312],
                        [ 0.0168, -0.0522, -0.0477],
                        [-0.0030, -0.0678, -0.0414],
                        [ 0.0070, -0.1418, -0.0216],
                        [ 0.0016, -0.0492, -0.0283],
                        [-0.0109, -0.0500, -0.0352],
                        [-0.0290, -0.0533, -0.0319]])


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

