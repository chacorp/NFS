import torch

expression_bases = [
    'browDown_L',
    'browDown_R',
    'browInnerUp_L',
    'browInnerUp_R',
    'browOuterUp_L',
    'browOuterUp_R',
    'cheekPuff_L',
    'cheekPuff_R',
    'cheekSquint_L',
    'cheekSquint_R',
    'eyeBlink_L',
    'eyeBlink_R',
    'eyeLookDown_L',
    'eyeLookDown_R', 
    'eyeLookIn_L', 
    'eyeLookIn_R', 
    'eyeLookOut_L',
    'eyeLookOut_R',
    'eyeLookUp_L',
    'eyeLookUp_R',
    'eyeSquint_L',
    'eyeSquint_R',
    'eyeWide_L',
    'eyeWide_R',
    'jawForward',
    'jawLeft',
    'jawOpen',
    'jawRight',
    'mouthClose',
    'mouthDimple_L',
    'mouthDimple_R',
    'mouthFrown_L',
    'mouthFrown_R',
    'mouthFunnel',
    'mouthLeft',
    'mouthLowerDown_L',
    'mouthLowerDown_R',
    'mouthPress_L',
    'mouthPress_R',
    'mouthPucker',
    'mouthRight',
    'mouthRollLower',
    'mouthRollUpper',
    'mouthShrugLower',
    'mouthShrugUpper',
    'mouthSmile_L',
    'mouthSmile_R', 
    'mouthStretch_L',
    'mouthStretch_R', 
    'mouthUpperUp_L', 
    'mouthUpperUp_R',
    'noseSneer_L',
    'noseSneer_R',
] 

ICT_KEYS = ['BrowDownLeft',
            'BrowDownRight',
            'BrowInnerUpLeft', 
            'BrowInnerUpRight', 
            'BrowOuterUpLeft', 
            'BrowOuterUpRight', 
            'CheekPuffLeft', 
            'CheekPuffRight', 
            'CheekSquintLeft', 
            'CheekSquintRight', 
            'EyeBlinkLeft', 
            'EyeBlinkRight', 
            'EyeLookDownLeft', 
            'EyeLookDownRight', 
            'EyeLookInLeft', 
            'EyeLookInRight', 
            'EyeLookOutLeft', 
            'EyeLookOutRight', 
            'EyeLookUpLeft', 
            'EyeLookUpRight', 
            'EyeSquintLeft', 
            'EyeSquintRight', 
            'EyeWideLeft', 
            'EyeWideRight', 
            'JawForward', 
            'JawLeft', 
            'JawOpen',
            'JawRight',
            'MouthClose',
            'MouthDimpleLeft',
            'MouthDimpleRight',
            'MouthFrownLeft',
            'MouthFrownRight',
            'MouthFunnel',
            'MouthLeft',
            'MouthLowerDownLeft', 
            'MouthLowerDownRight',
            'MouthPressLeft',
            'MouthPressRight',
            'MouthPucker',
            'MouthRight',
            'MouthRollLower',
            'MouthRollUpper',
            'MouthShrugLower',
            'MouthShrugUpper',
            'MouthSmileLeft',
            'MouthSmileRight',
            'MouthStretchLeft',
            'MouthStretchRight',
            'MouthUpperUpLeft',
            'MouthUpperUpRight',
            'NoseSneerLeft',
            'NoseSneerRight'
    ]

DATA_KEYS = ['Timecode', 'BlendshapeCount', 'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch', 'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll']

KEYS = ['BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight', 'EyeBlinkLeft', 'EyeBlinkRight', 'EyeLookDownLeft', 'EyeLookDownRight', 'EyeLookInLeft', 'EyeLookInRight', 'EyeLookOutLeft', 'EyeLookOutRight', 'EyeLookUpLeft', 'EyeLookUpRight', 'EyeSquintLeft', 'EyeSquintRight', 'EyeWideLeft', 'EyeWideRight', 'JawForward', 'JawLeft', 'JawOpen', 'JawRight', 'MouthClose', 'MouthDimpleLeft', 'MouthDimpleRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthFunnel', 'MouthLeft', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthPressLeft', 'MouthPressRight', 'MouthPucker', 'MouthRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper', 'MouthSmileLeft', 'MouthSmileRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthUpperUpLeft', 'MouthUpperUpRight', 'NoseSneerLeft', 'NoseSneerRight']


ict_data_synth = {
    'train': [f'{i:03d}' for i in range(201)], 
    'val':   [f'{i:03d}' for i in range(100)], 
    'test':  [f'{i:03d}' for i in range(100)], 
}

ict_data_split = {
    'train': ['m00', 'm02', 'm04', 'm05', 'm06', 'w00', 'w01', 'w02', 'w05', 'w08'], # 00 ~ 60
    'val':   ['m00', 'm02', 'm04', 'm05', 'm06', 'w00', 'w01', 'w02', 'w05', 'w08'], # 60 ~ 65
    'test':  ['m00', 'm02', 'm04', 'm05', 'm06', 'w00', 'w01', 'w02', 'w05', 'w08'], # 66 ~ 72
}
voca_data_split = {
    'train': ['FaceTalk_170728_03272_TA', 'FaceTalk_170904_00128_TA', 
              'FaceTalk_170725_00137_TA', 'FaceTalk_170915_00223_TA', 
              'FaceTalk_170811_03274_TA', 'FaceTalk_170913_03279_TA', 
              'FaceTalk_170904_03276_TA', 'FaceTalk_170912_03278_TA'],
    'val':   ['FaceTalk_170811_03275_TA', 'FaceTalk_170908_03277_TA'],
    'test':  ['FaceTalk_170809_00138_TA', 'FaceTalk_170731_00024_TA'],
}
biwi_data_split = {
    'train': ['F2', 'F3', 'F4', 'M3', 'M4', 'M5'],
    'val':   ['F2', 'F3', 'F4', 'M3', 'M4', 'M5'],
    'test':  ['F2', 'F3', 'F4', 'M3', 'M4', 'M5'], # if A set
    'test-B':['F1', 'F5', 'F6', 'F7', 'F8', 'M1', 'M2', 'M6'], # if B set
}
mf_data_split = {
    'train': ['m--20171024--0000--002757580--GHS', 'm--20180105--0000--002539136--GHS',
              'm--20180226--0000--6674443--GHS',   'm--20180227--0000--6795937--GHS',
              'm--20180406--0000--8870559--GHS',   'm--20180418--0000--2183941--GHS',
              'm--20180426--0000--002643814--GHS', 'm--20180510--0000--5372021--GHS',
              'm--20180927--0000--7889059--GHS',   'm--20181017--0000--002914589--GHS',
              'm--20190529--1004--5067077--GHS'],
    'val':   ['m--20190529--1300--002421669--GHS'], 
    'test':  ['m--20190828--1318--002645310--GHS']
}

def get_data_splits():
    return ict_data_split, voca_data_split, biwi_data_split, mf_data_split, ict_data_synth

def get_identity_num():
    identity_list = ict_data_split['train'] \
            + voca_data_split['train'] \
            + biwi_data_split['train'] \
            + mf_data_split['train']

    for mode in ['val', 'test']:
        identity_list += voca_data_split[mode] + mf_data_split[mode]

    identity_list += biwi_data_split['test-B']

    identity_num = {}
    for n, id_name in enumerate(identity_list):
        identity_num[id_name] = n
        
    return identity_num
