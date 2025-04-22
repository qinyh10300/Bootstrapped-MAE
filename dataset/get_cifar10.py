import os
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial

def save_image(example, split, classes, output_directory):
    """
    保存单张图片到对应的目录

    参数:
        example (dict): 数据集中的单个样本
        split (str): 数据集划分 ('train' 或 'test')
        classes (list): 类别名称列表
        output_directory (str): 输出的根目录

    返回:
        str: 保存的图片路径
    """
    # 获取图片和标签
    img = example['img']
    label = example['label']
    class_name = classes[label]
    
    # 确定类别对应的输出目录
    class_output_dir = os.path.join(output_directory, class_name)
    
    # 如果目录不存在，则创建
    os.makedirs(class_output_dir, exist_ok=True)
    
    # 生成唯一的文件名
    img_path = os.path.join(class_output_dir, f'{class_name}_{split}_{example["idx"]}.png')
    
    # 保存图片
    img.save(img_path)
    
    return img_path

def download_and_prepare_cifar10(root_dir, num_workers=None):
    """
    下载 CIFAR-10 数据集并保存为 ImageFolder 格式

    参数:
        root_dir (str): 保存数据集的根目录
        num_workers (int, optional): 多进程的工作线程数，默认为 None（使用所有可用核心）
    """
    # 创建训练集和测试集的目录
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # CIFAR-10 的类别名称
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 从 Hugging Face 下载数据集
    dataset = load_dataset('uoft-cs/cifar10', num_proc=24)

    # 为数据集添加索引列，用于生成唯一文件名
    for split in ['train', 'test']:
        dataset[split] = dataset[split].add_column('idx', range(len(dataset[split])))

    # 定义处理数据集划分的函数
    def process_split(split):
        """
        处理指定的数据集划分（训练集或测试集）

        参数:
            split (str): 数据集划分 ('train' 或 'test')
        """
        # 确定输出目录
        output_directory = train_dir if split == 'train' else test_dir
        
        # 使用 partial 函数绑定参数
        save_func = partial(
            save_image, 
            split=split, 
            classes=classes, 
            output_directory=output_directory
        )
        
        # 使用多进程保存图片
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用 tqdm 显示进度条
            list(tqdm(
                pool.imap(save_func, dataset[split]), 
                total=len(dataset[split]), 
                desc=f'保存 {split} 图片中'
            ))

    # 保存训练集和测试集图片
    process_split('train')
    process_split('test')

    print(f"CIFAR-10 数据集已下载并保存到 {root_dir}")

# 示例用法
if __name__ == "__main__":
    # 指定保存数据集的根目录
    root_dir = './dataset/cifar10_dataset'
    
    # 下载并准备数据集
    # 可以指定工作线程数，或者设置为 None 使用所有核心
    download_and_prepare_cifar10(root_dir, num_workers=None)

    # 验证数据集结构
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # 定义简单的变换（可选）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 使用 ImageFolder 加载数据集
    dataset_train = datasets.ImageFolder(
        os.path.join(root_dir, 'train'), 
        transform=transform
    )

    # 打印数据集信息
    print(f"训练集总图片数: {len(dataset_train)}")
    print(f"类别: {dataset_train.classes}")