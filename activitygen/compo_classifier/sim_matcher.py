from os.path import join as pjoin
import faiss
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import numpy as np
import json
import os

from activitygen.merge.element import Element


class TemplateDataset(Dataset):
    def __init__(self, template_dir_path, transform=None):
        self.template_dir_path = template_dir_path
        self.transform = transform
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def load_data(self):
        metadata_path = os.path.join(self.template_dir_path, "metadata.jsonl")
        data = []
        with open(metadata_path, "r") as file:
            for line in file:
                template_info = json.loads(line)
                data.append(template_info)
        return data

    def __getitem__(self, idx):
        template_info = self.data[idx]
        template_image_path = os.path.join(
            self.template_dir_path, template_info["template_file_name"]
        )
        template_image = Image.open(template_image_path).convert("RGB")

        if self.transform:
            template_image = self.transform(template_image)

        return template_image, template_info


class SimMatcher:
    def __init__(self, weight_path, template_dir_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 128
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.simclr_model = self.load_simclr_model(weight_path)
        self.template_dir_path = template_dir_path
        print("Building ViSM index...")
        self.template_dataset = TemplateDataset(
            template_dir_path, transform=self.transform
        )
        self.template_loader = DataLoader(
            self.template_dataset, batch_size=1, shuffle=False
        )
        self.template_embeddings, self.template_infos = self.build_template_data()
        self.template_index = self.build_faiss_index(self.template_embeddings)
        print("Done building ViSM index.")

    def load_simclr_model(self, weight_path):
        simclr_model = SimCLRModel(weight_path)
        simclr_model.to(self.device)
        simclr_model.eval()

        return simclr_model

    def build_template_data(self):
        embeddings = []
        template_infos = []
        with torch.no_grad():
            self.simclr_model.eval()
            for i, (images, infos) in enumerate(self.template_loader, 0):
                images = images.to(self.device)
                embeddings.append(self.simclr_model(images))
                template_infos.append(infos)

        embeddings = torch.cat(embeddings)
        embeddings = normalize(embeddings.cpu().numpy(), axis=1)

        return embeddings, template_infos

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def return_checkbox_state(self, query_checkbox):
        query_image = Image.fromarray(np.uint8(query_checkbox)).convert("RGB")
        query_image = self.transform(query_image).unsqueeze(0).to(self.device)
        query_embedding = self.simclr_model(query_image).detach().cpu().numpy()
        query_embedding = normalize(query_embedding, axis=1)

        dataset = datasets.ImageFolder(
            root=os.path.dirname(os.path.abspath(__file__))
            + "/predefined_templates/checkbox/",
            transform=self.transform,
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        embeddings = []
        image_paths = []
        with torch.no_grad():
            self.simclr_model.eval()
            for i, (images, _) in enumerate(data_loader):
                images = images.to(self.device)
                embeddings.append(self.simclr_model(images))
                image_paths.append(dataset.imgs[i][0])

        embeddings = torch.cat(embeddings)
        embeddings = normalize(embeddings.cpu().numpy(), axis=1)
        checbox_index = faiss.IndexFlatIP(embeddings.shape[1])
        checbox_index.add(embeddings)

        D, I = checbox_index.search(query_embedding, k=1)

        most_similar_indices = I[0]
        cosine_similarities = D[0]
        similar_image_paths = [image_paths[i] for i in most_similar_indices]

        return similar_image_paths, cosine_similarities

    def detect_template_matches(self, query_image_np, element: Element):
        query_image = Image.fromarray(np.uint8(query_image_np)).convert("RGB")
        query_image = self.transform(query_image).unsqueeze(0).to(self.device)
        query_embedding = self.simclr_model(query_image).detach().cpu().numpy()
        query_embedding = normalize(query_embedding, axis=1)

        D, I = self.template_index.search(query_embedding, k=1)

        most_similar_idx = I[0][0]
        cosine_sim = D[0][0]
        most_similar_template_info = self.template_infos[most_similar_idx]
        if "bbox" in most_similar_template_info:

            def are_bboxes_intersecting(template_bbox, elem):
                if (
                    template_bbox["x1"] < elem.x2
                    and template_bbox["x2"] > elem.x1
                    and template_bbox["y1"] < elem.y2
                    and template_bbox["y2"] > elem.y1
                ):
                    return True
                else:
                    return False

            bbox = most_similar_template_info["bbox"]
            if not are_bboxes_intersecting(bbox, element):
                cosine_sim = -2
        activity_name = "None"
        if (
            cosine_sim < most_similar_template_info["threshold"].detach().cpu().numpy()
            or cosine_sim == -2
        ):
            most_similar_template_name = None
        else:
            most_similar_template_name = str(
                most_similar_template_info["template_name"][0]
            )
            if "activity_name" in most_similar_template_info:
                activity_name = str(most_similar_template_info["activity_name"][0])
        return most_similar_template_name, activity_name


class SimCLRModel(nn.Module):
    def __init__(self, weight_path):
        super(SimCLRModel, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=False)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        if weight_path:
            ckpt = torch.load(weight_path)
            self.backbone.load_state_dict(ckpt["resnet50_parameters"])

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)  # Todo: Check if flatten is needed
        return y
