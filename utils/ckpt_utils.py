
def del_key(_dict, _key_word):
    del_keys = []
    for k, v in _dict.items():
        for d_ in _key_word:
            if d_ in k:
                del_keys.append(k)
    for dk in del_keys:
        del _dict[dk]
    return _dict

def replace_key(_dict, key_pairs):
    new_dict = {}
    for key in _dict.keys():
        # new_key = key.replace('module.encoder', 'id_encoder')
        for kp in key_pairs.keys():
            if kp in key:
                new_key = key.replace(kp, key_pairs[kp])
        new_dict[new_key] = _dict[key]
    return new_dict

def curl_key(_dict, _key_word):
    del_keys = []
    for k, v in _dict.items():
        for d_ in _key_word:
            if d_ not in k:
                del_keys.append(k)
    for dk in del_keys:
        del _dict[dk]
    return _dict

def reducename_key(_dict, key_pairs):
    new_dict = {}
    for key in _dict.keys():
        # new_key = key.replace('module.encoder', 'id_encoder')
        for kp in key_pairs:
            len_k = len(kp)
            if kp in key:
                new_key = key[len_k:]
            else:
                new_key = key
        new_dict[new_key] = _dict[key]
    return new_dict