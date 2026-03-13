import os
import torch
from .dataset import BaseDataset, ConditionalDataset
import json
config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace/config.json")
with open(config_file, "r") as f:
    running_config = json.load(f)
test_gpu_ids = running_config["test_gpu_ids"]




class ImageNet_ResNet18(BaseDataset):
    data_path = "./dataset/imagenet_resnet18/checkpoint"
    generated_path = "./dataset/imagenet_resnet18/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_resnet18/test.py " + \
                   "./dataset/imagenet_resnet18/generated/generated_model.pth"

class ImageNet_ResNet50(BaseDataset):
    data_path = "./dataset/imagenet_resnet50/checkpoint"
    generated_path = "./dataset/imagenet_resnet50/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_resnet50/test.py " + \
                   "./dataset/imagenet_resnet50/generated/generated_model.pth"

class ImageNet_ViTTiny(BaseDataset):
    data_path = "./dataset/imagenet_vittiny/checkpoint"
    generated_path = "./dataset/imagenet_vittiny/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_vittiny/test.py " + \
                   "./dataset/imagenet_vittiny/generated/generated_model.pth"

class ImageNet_ViTSmall(BaseDataset):
    data_path = "./dataset/imagenet_vitsmall/checkpoint"
    generated_path = "./dataset/imagenet_vitsmall/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_vitsmall/test.py " + \
                   "./dataset/imagenet_vitsmall/generated/generated_model.pth"

class ImageNet_ViTBase(BaseDataset):
    data_path = "./dataset/imagenet_vitbase/checkpoint"
    generated_path = "./dataset/imagenet_vitbase/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_vitbase/test.py " + \
                   "./dataset/imagenet_vitbase/generated/generated_model.pth"

class ImageNet_ConvNextAtto(BaseDataset):
    data_path = "./dataset/imagenet_convnextatto/checkpoint"
    generated_path = "./dataset/imagenet_convnextatto/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_convnextatto/test.py " + \
                   "./dataset/imagenet_convnextatto/generated/generated_model.pth"

class ImageNet_ConvNextLarge(BaseDataset):
    data_path = "./dataset/imagenet_convnextlarge/checkpoint"
    generated_path = "./dataset/imagenet_convnextlarge/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/imagenet_convnextlarge/test.py " + \
                   "./dataset/imagenet_convnextlarge/generated/generated_model.pth"

class CocoDetection(BaseDataset):
    data_path = "./dataset/downtask_detection/checkpoint"
    generated_path = "./dataset/downtask_detection/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_detection/test.sh " + \
    #                "./dataset/downtask_detection/generated/generated_model.pth"

class ADE20KSegmentation(BaseDataset):
    data_path = "./dataset/downtask_segmentation/checkpoint"
    generated_path = "./dataset/downtask_segmentation/generated/generated_model.pth"
    test_command = "echo \"Code for testing is coming soon!\n\""
    # test_command = "bash ./dataset/downtask_segmentation/test.sh " + \
    #                "./dataset/downtask_segmentation/generated/generated_model.pth"

class DoRACommonSenseReasoningR4(BaseDataset):
    data_path = "./dataset/downtask_dora_r4/checkpoint"
    generated_path = "./dataset/downtask_dora_r4/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/downtask_dora_r4/test.py " + \
                   "./dataset/downtask_dora_r4/generated/generated_model.pth"

class DoRACommonSenseReasoningR16(BaseDataset):
    data_path = "./dataset/downtask_dora_r16/checkpoint"
    generated_path = "./dataset/downtask_dora_r16/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/downtask_dora_r16/test.py " + \
                   "./dataset/downtask_dora_r16/generated/generated_model.pth"

class DoRACommonSenseReasoningR64(BaseDataset):
    data_path = "./dataset/downtask_dora_r64/checkpoint"
    generated_path = "./dataset/downtask_dora_r64/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/downtask_dora_r64/test.py " + \
                   "./dataset/downtask_dora_r64/generated/generated_model.pth"

class Cifar10_ResNet18(BaseDataset):
    data_path = "./dataset/cifar10_resnet18/checkpoint"
    generated_path = "./dataset/cifar10_resnet18/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar10_resnet18/test.py " + \
                   "./dataset/cifar10_resnet18/generated/generated_model.pth"

class Cifar10_MobileNetv3(BaseDataset):
    data_path = "./dataset/cifar10_mobilenetv3/checkpoint"
    generated_path = "./dataset/cifar10_mobilenetv3/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar10_mobilenetv3/test.py " + \
                   "./dataset/cifar10_mobilenetv3/generated/generated_model.pth"

class Cifar10_ViTBase(BaseDataset):
    data_path = "./dataset/cifar10_vitbase/checkpoint"
    generated_path = "./dataset/cifar10_vitbase/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar10_vitbase/test.py " + \
                   "./dataset/cifar10_vitbase/generated/generated_model.pth"

class Cifar10_CNNSmall(BaseDataset):
    data_path = "./dataset/cifar10_cnnsmall/checkpoint"
    generated_path = "./dataset/cifar10_cnnsmall/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar10_cnnsmall/test.py " + \
                   "./dataset/cifar10_cnnsmall/generated/generated_model.pth"

class Cifar10_CNNMedium(BaseDataset):
    data_path = "./dataset/cifar10_cnnmedium/checkpoint"
    generated_path = "./dataset/cifar10_cnnmedium/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar10_cnnmedium/test.py " + \
                   "./dataset/cifar10_cnnmedium/generated/generated_model.pth"

class Cifar100_ResNet18BN(BaseDataset):
    data_path = "./dataset/cifar100_resnet18bn/checkpoint"
    generated_path = "./dataset/cifar100_resnet18bn/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/cifar100_resnet18bn/test.py " + \
                   "./dataset/cifar100_resnet18bn/generated/generated_model.pth"




class Permutation_ViTTiny(ConditionalDataset):
    data_path = "./dataset/condition_permutation_vittiny/checkpoint"
    generated_path = "./dataset/condition_permutation_vittiny/generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/condition_permutation_vittiny/test.py " + \
                   "./dataset/condition_permutation_vittiny/generated/generated_model.pth"

    def _extract_condition(self, index: int):
        condition = super()._extract_condition(index)[2][5:]
        return int(condition)




class ClassInput_ViTTiny(ConditionalDataset):
    def _extract_condition(self, index: int):
        condition = super()._extract_condition(index)[2][5:]
        one_hot_string = bin(int(condition))[2:].zfill(10)
        optimize_class = [index for index, i in enumerate(one_hot_string) if i == "1"]
        indicator_tensor = torch.zeros(size=(10,))
        for i in optimize_class:
            indicator_tensor[i] = 1.0
        return indicator_tensor

class ClassInput_ViTTiny_Train(ClassInput_ViTTiny):
    data_path = "./dataset/condition_classinput_vittiny/checkpoint_train"
    generated_path = None
    test_command = None

class ClassInput_ViTTiny_Test(ClassInput_ViTTiny):
    data_path = "./dataset/condition_classinput_vittiny/checkpoint_test"
    generated_path = "./dataset/condition_classinput_vittiny/generated/generated_model_class{}.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./dataset/condition_classinput_vittiny/test.py " + \
                   "./dataset/condition_classinput_vittiny/generated/generated_model_class{}.pth"






# #################################### user-defined dataset classes here ####################################

class BipedalWalker_PPO(ConditionalDataset):
    data_path = "./test_data/bipedal_walker_data/bipedal_walker"
    generated_path = "./experiments/bipedal_walker/test_generated/generated_walker.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./experiments/bipedal_walker/test.py " + \
                   "./experiments/bipedal_walker/test_generated/generated_walker.pth"

    def _extract_condition(self, index: int):
        """Extract [leg_length, leg_width, gravity, friction] from filename."""
        from experiments.bipedal_walker.env_utils import parse_params_from_filename
        filename = os.path.basename(self.checkpoint_list[index])
        params = parse_params_from_filename(filename)
        return torch.tensor([
            params["leg_length"],
            params["leg_width"],
            params["gravity"],
            params["friction"]
        ], dtype=torch.float32)


class LunarLander_PPO(ConditionalDataset):
    data_path = "./test_data/lunar_lander_data/lunar_lander"
    generated_path = "./experiments/lunar_lander/test_generated/generated_lander.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./experiments/lunar_lander/test.py " + \
                   "./experiments/lunar_lander/test_generated/generated_lander.pth"

    def _extract_condition(self, index: int):
        """Extract [w_landing, w_fuel, w_time, w_smoothness] from filename."""
        from experiments.lunar_lander.test import parse_weights_from_filename
        filename = os.path.basename(self.checkpoint_list[index])
        params = parse_weights_from_filename(filename)
        return torch.tensor([
            params["w_landing"],
            params["w_fuel"],
            params["w_time"],
            params["w_smoothness"]
        ], dtype=torch.float32)

    def __getitem__(self, index):
        """Override to detach loaded tensors (lunar lander checkpoints save raw nn.Parameters)."""
        index = index % self.real_length
        diction = torch.load(self.checkpoint_list[index], map_location="cpu", weights_only=True)
        diction = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in diction.items()}
        condition = self._extract_condition(index)
        param = self.preprocess(diction)
        return param.detach(), condition, index

    def get_structure(self):
        """Override to detach nn.Parameter tensors before computing structure stats."""
        checkpoint_list = self.checkpoint_list
        structures = [{} for _ in range(len(checkpoint_list))]
        for i, checkpoint in enumerate(checkpoint_list):
            diction = torch.load(checkpoint, map_location="cpu", weights_only=True)
            diction = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in diction.items()}
            for key, value in diction.items():
                if ("num_batches_tracked" in key) or (value.numel() == 1) or not torch.is_floating_point(value):
                    structures[i][key] = (value.shape, value, None)
                elif "running_var" in key:
                    pre_mean = value.mean() * 0.95
                    value = torch.log(value / pre_mean + 0.05)
                    structures[i][key] = (value.shape, pre_mean, value.mean(), value.std())
                else:  # conv & linear
                    structures[i][key] = (value.shape, value.mean(), value.std())
        final_structure = {}
        structure_diction = torch.load(checkpoint_list[0], map_location="cpu", weights_only=True)
        structure_diction = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in structure_diction.items()}
        for key, param in structure_diction.items():
            if ("num_batches_tracked" in key) or (param.numel() == 1) or not torch.is_floating_point(param):
                final_structure[key] = (param.shape, param, None)
            elif "running_var" in key:
                value = [param.shape, 0., 0., 0.]
                for structure in structures:
                    for i in [1, 2, 3]:
                        value[i] += structure[key][i]
                for i in [1, 2, 3]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
            else:  # conv & linear
                value = [param.shape, 0., 0.]
                for structure in structures:
                    for i in [1, 2]:
                        value[i] += structure[key][i]
                for i in [1, 2]:
                    value[i] /= len(structures)
                final_structure[key] = tuple(value)
        self.structure = final_structure
        return self.structure


class MLP_Regression(ConditionalDataset):
    data_path = "./experiments/mlp_regression/dataset"
    generated_path = "./experiments/mlp_regression/test_generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./experiments/mlp_regression/test.py " + \
                   "./experiments/mlp_regression/test_generated/generated_model.pth"

    def _extract_condition(self, index: int):
        """Extract [a, b, f, phi] from filename like func_a1.50_b0.30_f2.00_p1.57.pth."""
        import re
        filename = os.path.basename(self.checkpoint_list[index])
        pattern = r"func_a([\d.-]+)_b([\d.-]+)_f([\d.-]+)_p([\d.-]+)\.pth"
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Cannot parse condition from filename: {filename}")
        a = float(match.group(1))
        b = float(match.group(2))
        f = float(match.group(3))
        phi = float(match.group(4))
        return torch.tensor([a, b, f, phi], dtype=torch.float32)


class MLP_Regression_Train(MLP_Regression):
    data_path = "./experiments/mlp_regression/dataset_train"
    generated_path = None
    test_command = None


class MLP_Regression_Val(MLP_Regression):
    data_path = "./experiments/mlp_regression/dataset_val"
    generated_path = "./experiments/mlp_regression/test_generated/generated_model.pth"
    test_command = f"CUDA_VISIBLE_DEVICES={test_gpu_ids} python ./experiments/mlp_regression/test.py " + \
                   "./experiments/mlp_regression/test_generated/generated_model.pth"


# #################################### user-defined dataset classes here ####################################