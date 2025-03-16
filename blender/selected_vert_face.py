import bpy

Mode = 0
if Mode:
    obj = bpy.context.active_object
    # if EDIT mode, print selected vertex indices and face indices in console window
    if obj.mode == 'EDIT':
        # update vertex info
        bpy.ops.object.mode_set(mode='OBJECT')
        
        v_idx_list = []
        f_idx_list = []
        
        selected_vertices = [v for v in obj.data.vertices if v.select]
        for v in selected_vertices:
            v_idx_list.append(v.index)
        
        selected_faces = [f for f in obj.data.polygons if f.select]
        for f in selected_faces:
            f_idx_list.append(f.index)
            
        print("selected_vertices=", v_idx_list)
        #print("selected_faces=",   f_idx_list)
        
        # return to edit mode
        bpy.ops.object.mode_set(mode='EDIT')
    else:
        print("The active object is not in edit mode.")


else:
    ############################# show vertex indices ############################
    obj = bpy.context.active_object

    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='DESELECT')

    bpy.ops.object.mode_set(mode='OBJECT')

    for idx in vertex_indices:
        if idx < len(obj.data.vertices):
            obj.data.vertices[idx].select = True

    bpy.ops.object.mode_set(mode='EDIT')