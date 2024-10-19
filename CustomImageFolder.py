# 0. Write a custom dataset class
from torch.utils.data import Dataset

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
  # 2.Initialize our custom dataset
  def __init__(self,
               targ_dir:str,
               transform=None):
    # 3.Create class attributes
    # Get all of the image paths
    self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
    # Setup transform
    self.transform = transform
    # Create Classes and class_to_idx attributes
    self.classes,self.class_to_idx = find_classes(targ_dir)

  # 4. Create a function to load images
  def load_image(self,index:int) -> Image.Image:
    "Opens an image via a path and returns it"
    image_path = self.paths[index]
    return Image.open(image_path)

  # 5. Overwrite __len__()
  def __len__(self) ->int:
    "Returns the total number of samples "

    return len(self.paths)

  # 6. Overwrite __getitem__() method to return a particular sample
  def __getitem__(self,index:int) -> Tuple[torch.Tensor,int]:
    "Returns one sample of data , data and label (X,y)"
    img=self.load_image(index)
    class_name = self.paths[index].parent.name # expects path in format : data_folder/class_name/image.jpg
    class_idx = self.class_to_idx[class_name]

    # Transform if necessary
    if self.transform:
      return (self.transform(img),class_idx) # return transformed image and label
    else:
      return (img,class_idx) # return untransformed img and label

# # Create a transform
# from torchvision import transforms
# train_transforms = transforms.Compose([
#     transforms.Resize(size=(64,64)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor()
# ])
# test_transforms = transforms.Compose([
#     transforms.Resize(size=(64,64)),
#     transforms.ToTensor()
# ])

# # Test out ImageFolderCustom
# train_data_custom= ImageFolderCustom(targ_dir=train_dir,
#                                      transform=train_transforms)
# test_data_custom= ImageFolderCustom(targ_dir=test_dir,
#                                      transform=test_transforms)
# train_data_custom,test_data_custom
