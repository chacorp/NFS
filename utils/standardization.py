import torch

def get_new_mesh(id_verts, id_faces, v_idx, invert=False):
    """
    """
    max_index = id_verts.shape[0]

    new_vertex_indices = torch.arange(max_index)

    if invert:
        mask = torch.zeros(max_index, dtype=torch.bool)
        mask[v_idx] = 1
    else:
        mask = torch.ones(max_index, dtype=torch.bool)
        mask[v_idx] = 0

    updated_verts     = id_verts[mask]
    updated_verts_idx = new_vertex_indices[mask]

    index_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(updated_verts_idx)}

    updated_faces = torch.tensor([
                    [index_mapping.get(idx.item(), -1) for idx in face]
                    for face in id_faces
                ])

    valid_faces = ~torch.any(updated_faces == -1, dim=1)
    updated_faces = updated_faces[valid_faces]
    return updated_verts, updated_faces, updated_verts_idx
