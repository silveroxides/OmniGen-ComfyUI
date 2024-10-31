# OmniGen-ComfyUI
a custom node for [OmniGen](https://github.com/VectorSpaceLab/OmniGen), you can find [workflow here](./doc/)

## EXample
in prompt text, you only need `image_1`, text will auto be `<img><|image_1|></img>`
|text|image_1|image_2|image_3|out_img|
|--|--|--|--|--|
|`A curly-haired man in a red shirt is drinking tea.`|--|--|--|![](./doc/ComfyUI_temp_mdplu_00001_.png)|
|`The woman in image_1 waves her hand happily in the crowd`|![](./doc/zhang.png)|--|--|![](./doc/ComfyUI_temp_pphmf_00001_.png)|

## Tips
For out of memory or time cost, you can refer to [inference.md#requiremented-resources](https://github.com/VectorSpaceLab/OmniGen/blob/main/docs/inference.md#requiremented-resources) to select a appropriate setting.

```
{"task_type":"text_to_iamge","instruction":"A white cat resting on a picnic table.","input_images":[],"output_image":"cat.png"}
{"task_type":"image_edit","instruction":"<img><|image_1|></img> The umbrella should be red.","input_images":["edit_source_1.png"],"output_image":"edit_target_1.png"}
{"task_type":"segementation","instruction":"Find lamp in the picture <img><|image_1|></img> and color them blue.","input_images":["seg_input.png"],"output_image":"seg_output.png"}
{"task_type":"try-on","instruction":"<img><|image_1|></img> wears <img><|image_2|></img>.","input_images":["model.png","clothes.png"],"output_image":"try_on.png"}
{"task_type":"pose", "instruction": "Detect the skeleton of human in <img><|image_1|></img>", "input_images": ["human_pose.png"], "output_image": "pose.png"}
```
