import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Baseline(nn.Module):
    def __init__(
        self,
        embedding_dim=512,
        bert_model="bert-base-uncased",
        freeze_bert=True,
        freeze_resnet=True,
        use_sentence_bert=True, # parametre per l'us de sentence bert
        fine_tune_resnet_tail = False, # parametre per fine tunning
        remove_first_sentence = False, # parametre per treure la primera frase
        gtr_model_name="sentence-transformers/gtr-t5-base",
        use_gtr=False
    ):
        super(Baseline, self).__init__()

        self.use_sentence_bert = use_sentence_bert
        self.bert = None  # Se cargará más adelante si es necesario
        self.use_gtr = use_gtr
        self.gtr_model_name = gtr_model_name
        self.fine_tune_resnet_tail = fine_tune_resnet_tail
        self.remove_first_sentence_flag = remove_first_sentence

        # TEXT ENCODER
        print(f"[CONFIG] use_sentence_bert = {self.use_sentence_bert}")
        if not self.use_sentence_bert and not self.use_gtr:
            print("[INFO] → Carregant BERT clàssic (transformers.BertModel)")
            self.bert = BertModel.from_pretrained(bert_model)
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
            if freeze_bert:
                print("[INFO] → BERT congelat (freeze_bert = True)")
                for param in self.bert.parameters():
                    param.requires_grad = False
            else:
                print("[INFO] → BERT entrenable (freeze_bert = False)")
            self.bert_output_dim = self.bert.config.hidden_size
            self.text_projection = nn.Sequential(
                nn.Linear(self.bert_output_dim, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, embedding_dim),
            )
        else:
            print("[INFO] → S'utilitzarà Sentence-BERT (carregat més tard)")
            self.bert_output_dim = 768  # o 384 si canvies el model real o 768 grt

        # IMAGE ENCODER
        print(f"[CONFIG] fine_tune_resnet_tail = {self.fine_tune_resnet_tail}")
        print(f"[CONFIG] freeze_resnet = {freeze_resnet}")
        resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_resnet:
            print("[INFO] → ResNet congelada (freeze_resnet = True)")
            for param in resnet_base.parameters():
                param.requires_grad = False
        else:
            print("[INFO] → ResNet entrenable (freeze_resnet = False)")

        if fine_tune_resnet_tail:
            print("[INFO] → Fine-tune activat per ResNet.layer4 + fc")
            for name, module in resnet_base.named_children():
                if name in ["layer4", "fc"]:
                    for param in module.parameters():
                        param.requires_grad = True


        # Després assigna self.resnet al model tallat (sense fc)
        self.resnet = nn.Sequential(*list(resnet_base.children())[:-1])

        self.resnet_output_dim = 2048
        self.image_projection = nn.Sequential(
            nn.Linear(self.resnet_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, embedding_dim),
        )

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load_sentence_bert(self):
        if self.bert is None:
            if self.use_gtr:
                print(f"[INFO] Loading GTR model: {self.gtr_model_name}")
                self.bert = SentenceTransformer(self.gtr_model_name)
            else:
                print("[INFO] Loading Sentence-BERT model...")
                self.bert = SentenceTransformer("all-MiniLM-L6-v2")
                #self.bert = SentenceTransformer("all-mpnet-base-v2")

    def remove_first_sentence(self, texts):
        processed = []
        for t in texts:
            sentences = t.strip().split(".")
            rest = ".".join(sentences[1:]).strip()
            if not rest:
                rest = sentences[0]
            processed.append(rest)
        return processed

    def preprocess_text(self, text_samples):
        return self.bert_tokenizer(
            text_samples,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    def preprocess_image(self, image_paths):
        images = []
        for img in image_paths:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(img, Image.Image):
                img = self.image_transform(img)
            images.append(img)
        return torch.stack(images)

    def encode_text(self, text_features=None, raw_text=None):
        if self.use_sentence_bert or self.use_gtr:
            self.load_sentence_bert()
            if raw_text is None:
                raise ValueError("Sentence-BERT necesita raw_text.")

            cleaned_text = raw_text

            if self.remove_first_sentence_flag:  # Eliminem la primera frase si aixi esta especificat
                cleaned_text = self.remove_first_sentence(raw_text)

            trimmed_text = [" ".join(t.strip().split()[:100]) for t in cleaned_text]

            text_embeddings = self.bert.encode(trimmed_text, convert_to_tensor=True).to(device)
        else:
            if raw_text is not None:
                text_features = self.preprocess_text(raw_text).to(device)
            with torch.no_grad() if not self.bert.training else torch.enable_grad():
                outputs = self.bert(**text_features)
                text_features = outputs.last_hidden_state[:, 0, :]  # CLS token
            text_embeddings = self.text_projection(text_features)

        return F.normalize(text_embeddings, p=2, dim=1)

    def encode_image(self, image_features=None, image_paths=None):
        if image_paths is not None:
            image_features = self.preprocess_image(image_paths).to(device)

        with torch.no_grad() if not self.resnet[0].training else torch.enable_grad():
            image_features = self.resnet(image_features)
            image_features = image_features.flatten(start_dim=1)

        image_embeddings = self.image_projection(image_features)
        return F.normalize(image_embeddings, p=2, dim=1)

    def forward(self, text_features=None, image_features=None, raw_text=None, image_paths=None):
        text_embeddings = None
        image_embeddings = None

        if text_features is not None or raw_text is not None:
            text_embeddings = self.encode_text(text_features, raw_text)

        if image_features is not None or image_paths is not None:
            image_embeddings = self.encode_image(image_features, image_paths)

        return text_embeddings, image_embeddings

    def compute_similarity(self, text_embeddings, image_embeddings):
        return torch.mm(text_embeddings, image_embeddings.t())
