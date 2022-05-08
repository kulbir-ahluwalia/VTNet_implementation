# loading the possible scenes
def loading_scene_list(phase):
    scenes = []

    for i in range(4):
        if phase == 'train':
            for j in range(20):
                if i == 0:
                    scenes.append("FloorPlan" + str(j + 1))
                else:
                    scenes.append("FloorPlan" + str(i + 1) + '%02d' % (j + 1))
        elif phase == 'eval':
            eval_scenes_list = []
            for j in range(10):
                if i == 0:
                    eval_scenes_list.append("FloorPlan" + str(j + 1))
                else:
                    eval_scenes_list.append("FloorPlan" + str(i + 1) + '%02d' % (j + 1 + 20))
            scenes.append(eval_scenes_list)

    return scenes