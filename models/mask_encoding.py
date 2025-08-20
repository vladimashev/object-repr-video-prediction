import torch
import torch.nn as nn

# MOVi-C has up to 10 objects + add one more for background
DEFAULT_OBJECTS_COUNT = 11

class MaskEncoder(nn.Module):
    def __init__(self, obj_num=DEFAULT_OBJECTS_COUNT):
        super().__init__()
        self.obj_num = obj_num
    
    def forward(self, item):
        img, masks = item
        objects = []

        img = img.float()      # [C, H, W]
        segmap = masks.long()  # [H, W], integer labels

        # get unique object IDs (exclude background=0)
        obj_ids = torch.unique(segmap)
        obj_ids = obj_ids[obj_ids != 0]

        # extract each object by ID
        for oid in obj_ids:
            mask = (segmap == oid).float()      # [H, W]
            masked_obj = img * mask.unsqueeze(0)  # [C, H, W]
            objects.append(masked_obj)

        # compute background separately
        BG_ID = 0
        bg_mask = (segmap == BG_ID).float()
        background = img * bg_mask.unsqueeze(0)
        objects.append(background)

        # Now enforce fixed number of objects
        num_objs = len(objects)
        if num_objs < self.obj_num:
            # add empty objects
            pad_objs = [torch.zeros_like(img) for _ in range(self.obj_num - num_objs)]
            objects.extend(pad_objs)
        # ?could there be more objects? remove then?
        elif num_objs > self.obj_num:
            objects = objects[:self.obj_num]

        objects = torch.stack(objects, dim=0)  # [obj_num, C, H, W]

        return objects


class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, objects):
        BG_ID = -1 # assume bg is the last obj in array
        
        # last object might be empty embedding,
        # iterate until first non-zero is met
        for i in reversed(range(objects.shape[0])):
            if objects[i].sum() > 0:
                BG_ID = i
                break
        background = objects[BG_ID]
        
        obj_slots = objects[:BG_ID]

        # TODO what if objects overlap? for now assume they are always disjoint
        return torch.clamp(obj_slots.sum(dim=0) + background, 0, 1)
