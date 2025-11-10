import requests
import pandas as pd
from tqdm import tqdm
import os
import time
import logging
import shutil
import sys
from tqdm.notebook import tqdm
from PIL import Image, ImageOps
from roboflow import Roboflow
import cv2
from typing import *
import random as rd
from sklearn.model_selection import train_test_split

class inaturqalistScrapper:
  """
    A class to scrape images from iNaturalist for a given scientific name or taxon ID.

    It fetches observations, extracts image URLs, and downloads the images.
    """

  def __init__(self, scientificName: str, n:int = 1000, taxon_id = "", save_folder_name= ""):
    """
    Scrape images from iNaturalist for a given scientific name or taxon ID.

    Parameters
    ----------
    scientificName : str
        The scientific name of the species to scrape images for.
    n : int, optional
        The number of images to scrape. Default is 1000.
    taxon_id : str, optional
        The taxon ID to use for scraping. If not provided, it will be fetched using the scientific name.
    save_folder_name : str, optional
        The name of the folder to save the images. If not provided, it will be set to the scientific name.
    """

    self.scientificName = scientificName
    self.n = n
    if not taxon_id:
      taxon_id, taxon_name = self.get_taxon_id()
    if not taxon_id:
        raise ValueError(f'No taxon found for scientific name: {scientificName}')

    observations = self.fetch_observations(taxon_id)
    image_urls = self.extract_image_urls(observations)[::-1]
    print(f'Scrapped {len(image_urls)} number of image links')

    self.download_images(image_urls, save_folder_name)

  def get_taxon_id(self):
      """
        Fetch the taxon ID for the given scientific name from iNaturalist API.

        Returns
        -------
        tuple
            A tuple containing the taxon ID and name if found, otherwise (None, None).
        """
      search_url = "https://api.inaturalist.org/v1/taxa"
      params = {
          'q': self.scientificName,
          'rank': 'species',
          'per_page': 1
      }
      try:
          response = requests.get(search_url, params=params)
          response.raise_for_status()
          data = response.json()
          if data['total_results'] > 0:
              taxon = data['results'][0]
              return taxon['id'], taxon['name']
          else:
              return None, None
      except requests.exceptions.RequestException as e:
          print(f"Error fetching taxon ID: {e}")
          return None, None

  def fetch_observations(self, taxon_id, per_page=10000):
      observations = []
      page = 1
      total_pages = 1000
      while True:
          print(f"Fetching page {page}...")
          url = "https://api.inaturalist.org/v1/observations"
          params = {
              'taxon_id': taxon_id,
              'per_page': per_page,
              'page': page,
              'order': 'desc',
              'order_by': 'created_at',
              'ident_taxon_id': taxon_id
          }
          try:
              response = requests.get(url, params=params)
              response.raise_for_status()
              data = response.json()
              results = data.get('results', [])
              meta = data.get('meta', {})
              if not results:
                  # No more observations
                  break
              observations.extend(results)
              print(f"Fetched {len(results)} observations.")
              if len(observations) >= self.n:
                  break
              if total_pages is None:
                  total_results = meta.get('total_results', 0)
                  total_pages = (total_results // per_page) + (1 if total_results % per_page != 0 else 0)
              if page > total_pages:
                  break
              page += 1
              time.sleep(1)  # Respect rate limits
          except requests.exceptions.RequestException as e:
              print(f"Failed to fetch observations: {e}")
              break
      return observations

  def extract_image_urls(self, observations):
      image_urls = []
      for obs in observations:
          photos = obs.get('photos', [])
          for photo in photos:
              url = photo.get('url', '')
              if url:
                  # Attempt to get 'original' and 'large' sizes
                  original_url = url.replace('square', 'original')
                  image_urls.append(original_url)
      # Remove duplicate URLs
      image_urls = list(set(image_urls))
      return image_urls

  def download_images(self, image_urls, save_folder_name="", max_retries=3):
    download_path = '../DATA/src_SCRAPPING/'
    if not save_folder_name: 
        save_folder_name = self.scientificName
    
    save_dir = os.path.join(download_path, save_folder_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
      shutil.rmtree(save_dir)
      os.makedirs(save_dir)
      
    # Set up logging for failed downloads
    logging.basicConfig(filename='!download_errors.log', level=logging.ERROR)

    index=0
    if self.n > len(image_urls): self.n = len(image_urls)
    for index_url in tqdm(range(self.n), desc="Downloading images"):
        filename = f'{index}.{image_urls[index_url].split(".")[-1]}'
        retries = 0
        success = False
        while retries < max_retries and not success:
            try:
                response = requests.get(image_urls[index_url], stream=True, timeout=10)
                if response.status_code == 200:
                    file_path = os.path.join(save_dir, filename)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            if chunk:
                                f.write(chunk)
                                index+=1
                    success = True
                else:
                    print(f"Failed to download {image_urls[index_url]}: Status code {response.status_code}")
                    retries += 1
                    time.sleep(2)  # Wait before retrying
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {image_urls[index_url]}: {e}")
                retries += 1
                time.sleep(2)  # Wait before retrying
        if not success:
            logging.error(f"Failed to download after {max_retries} retries: {image_urls[index_url]}")

def check_all_images_in_folder(folder_path):
  '''
  Check all images in a folder and return a DataFrame with the folder name and number of images.

  Parameters
  ----------
  folder_path : str
    path to the folder with images

  Returns
  ---------
  pd.DataFrame: DataFrame 
    Dataframe with columns ["Folder name", "Len(Images in folder)"]
  '''
  if not os.path.exists(folder_path):
    raise ValueError("Folder path does not exist")
  image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff'}
  idklist = []
  for folder in tqdm(os.listdir(folder_path)):
    images = [i for i in os.listdir(os.path.join(folder_path, folder)) if i.split(".")[-1].lower() in image_extensions]
    idklist.append([folder, len(images)])
  return pd.DataFrame(idklist, columns=["Folder name", "Len"])



def crop_image(path, max_display_size=800):
    '''
    Crop an image interactively using OpenCV.

    Parameters
    ----------
    path : str
        The path to the image file to be cropped.
    max_display_size : int, optional
        The maximum size for displaying the image. Default is 800 pixels.
        
    Returns
    -------
    cropped_image : np.ndarray or None
        The cropped image as a NumPy array, or None if no cropping was done.
    '''
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Could not load image {path}")
        return None

    orig_height, orig_width = image.shape[:2]
    scale = min(max_display_size / orig_width, max_display_size / orig_height) if orig_width > max_display_size or orig_height > max_display_size else 1.0
    display_image = cv2.resize(image, (int(orig_width * scale), int(orig_height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()
    
    r = cv2.selectROI("Crop Image - Press ENTER or SPACE when done, 'c' to cancel", display_image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Crop Image - Press ENTER or SPACE when done, 'c' to cancel")
    
    if r[2] == 0 or r[3] == 0:
        print("No ROI selected. Skipping cropping.")
        return None
    
    x, y, w, h = [int(v / scale) for v in r]
    return image[y:y+h, x:x+w]

def start_cropping(input_folder, output_folder):
    '''
    Start the cropping process for images in a folder.

    Parameters
    ----------
    input_folder : str
        The folder containing images to crop.
    output_folder : str
        The folder where cropped images will be saved.
    '''
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        if os.path.isfile(input_path) and filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            cropped_img = crop_image(input_path)
            if cropped_img is not None:
                cv2.imwrite(output_path, cropped_img)
                print(f"Saved cropped image: {output_path}")
            else:
                print(f"Skipping {filename}")


def startDataCleaning(input_dir_name, output_dir_name, resize_percent=0.7):
    '''
    Start the data cleaning process by converting images to .jpg format, renaming, resolution downgrade, and moving them to a new directory
    
    Parameters
    ----------
    output_dir_name : str
        The name of the output directory where cleaned images will be stored.
    resize_percent : float, optional
        The percentage to resize the images if file exceeds 2mb. Default is 0.7 (70% of original size).
    '''
    OUTPUT_DIR = output_dir_name

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        raise ValueError(f"Directory {OUTPUT_DIR} already exists. Are you sure this is the correct name? Delete it first.")
    MENTAH_FULL_PATH = input_dir_name

    for theClass in tqdm(os.listdir(MENTAH_FULL_PATH), desc="Folders "):
        index = 0
        for img_name in tqdm(os.listdir(os.path.join(MENTAH_FULL_PATH, theClass)), desc=f"Images in {theClass}", leave=True):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(MENTAH_FULL_PATH, theClass, img_name)

                try:
                    img = Image.open(image_path)
                    img = ImageOps.exif_transpose(img)
                    width, height = img.size

                    # If file exceeds 2mb, resize it
                    if os.path.getsize(image_path) > 2 * 1024 * 1024:  # 2 MB in bytes
                        new_width = int(width * resize_percent)
                        new_height = int(height * resize_percent)
                        img = img.resize((new_width, new_height))


                    # Construct the output path
                    os.makedirs(os.path.join(OUTPUT_DIR, theClass), exist_ok=True)

                    new_filename = f"{theClass}-{index}{os.path.splitext(img_name)[1]}"
                    new_filepath = os.path.join(OUTPUT_DIR, theClass, new_filename)

                    img.convert("RGB").save(new_filepath, "JPEG")

                    index+=1

                except IOError as e:
                    print(f"Error processing {image_path}: {e}")

def unPackData_fromYOLO(input_dir, output_dir, keymapping):
    '''
    Unpacks image and label files from YOLO format to a structured directory.
    Creates new image based from the annotations and saves them in the output directory.

    Parameters
    ----------
    input : str
        Path to the input directory containing YOLO formatted images and labels.
    output : str
        Path to the output directory where unpacked images and labels will be saved.
    keymapping : dict
        A dictionary mapping YOLO class indices to human-readable class names.
        Only the keys of the dictionary will be used to create subdirectories in the output folder.
    '''
    input_imageDIR = os.path.join(input_dir, 'images')
    input_labelDIR = os.path.join(input_dir, 'labels')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in tqdm(os.listdir(input_imageDIR)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_imageDIR, filename)
            label_path = os.path.join(input_labelDIR, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            if not os.path.exists(label_path):
                print(f"Label file for {filename} does not exist. Skipping.")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            img = Image.open(image_path)
            width, height = img.size

            for line in lines:
                parts = line.strip().split()
                class_index = int(parts[0])
                if class_index not in keymapping.keys():
                    continue
                x_center, y_center, w, h = map(float, parts[1:])

                # Convert YOLO format to pixel coordinates
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                w_px = int(w * width)
                h_px = int(h * height)

                # Calculate bounding box coordinates
                x1 = max(0, x_center_px - w_px // 2)
                y1 = max(0, y_center_px - h_px // 2)
                x2 = min(width, x_center_px + w_px // 2)
                y2 = min(height, y_center_px + h_px // 2)

                # Crop the image
                cropped_img = img.crop((x1, y1, x2, y2))

                # Save the cropped image in the corresponding class directory
                output_class_dir = os.path.join(output_dir, keymapping[class_index])
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                output_image_path = os.path.join(output_class_dir, filename)
                cropped_img.save(output_image_path)
    
# Optional: If you want to use Roboflow for uploading datasets
class Upload_Roboflow():
    '''
    Uploads a dataset to Roboflow using the Roboflow API.
    '''
    def __init__(self, api_key: str, workspace:str, project:str):
        '''
        Initializes the Upload_Roboflow class with the given API key, workspace, and project name.

        Parameters
        ------------
        - api_key: str
            Roboflow API key.
        - workspace: str
            Roboflow workspace name.
        - project: str
            Roboflow project name.
        '''
        try:
            self.rf = Roboflow(api_key=api_key)
        except Exception as e:
            raise Exception(f"Wrong API key: {e}")
        try:
            self.project = self.rf.workspace(workspace).project(project)
        except Exception as e:
            raise Exception(f"Wrong workspace or project name: {e}")
    
    def upload_dir_image(self, img_dir: str):
        '''
        Uploads images from a directo ry to the Roboflow project.
        
        Parameters
        ------------
        - img_dir: str
            Path to the directory containing images to upload.
        '''
        for image_path in tqdm(os.listdir(img_dir)):
            self.project.upload(
                image_path=image_path,
                # batch_name="YOUR_BATCH_NAME",
                # split="train",
                num_retry_uploads=3,
                # tag_names=["YOUR_TAG_NAME"],
                # sequence_number=99,
                # sequence_size=100
        )

def split_equally(src, dst, composition = [0.9, 0.1], seed=rd.randint(0, 10000)):
    """
    Splits the images in the source directory into two directories based on the given composition.
    The split is done randomly, and the seed is used for reproducibility.

    Parameters
    ------------
    - src: 
        The source directory containing the images to be split.
    - dst: 
        The destination directory where the split images will be saved.
    - composition:
        A list of two floats representing the proportions of the dataset to allocate to train and test sets. 
        The values should sum to 1 (e.g., [0.9, 0.1]).
    - seed:
        Random seed for reproducibility.
    """
    temp = {}
    assert sum(composition) == 1, "Composition must sum to 1"
    for class_name in tqdm(os.listdir(f'../DATA/{src}'), desc="Splitting data into (train+eval) and test sets"):
        class_path = os.path.join(f'../DATA/{src}', class_name)

        # Create paths for train+eval and test directories
        for split in ["train_val", "test"]:
            os.makedirs(os.path.join("../DATA/", dst, split, class_name), exist_ok=True)

        train_image, test_image = train_test_split(os.listdir(class_path), test_size=composition[1], random_state=seed)

        for image in train_image:
            shutil.copy(os.path.join(class_path, image), os.path.join("../DATA/", dst, "train_val", class_name, image))
        for image in test_image:
            shutil.copy(os.path.join(class_path, image), os.path.join("../DATA/", dst, "test", class_name, image))
        temp[class_name] = {
            "train_val": len(train_image),
            "test": len(test_image)
        }
    df = pd.DataFrame(temp).T
    df.index.name = "Class"
    df.reset_index(inplace=True)
    df['train_val'] = df['train_val'].astype(int)
    df['test'] = df['test'].astype(int)
    return df
