
from typing import Tuple
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from app.utils.logger import get_logger
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


logger = get_logger(__name__)


def _estimate_confidence(outputs) -> float:
    scores = getattr(outputs, "scores", None)
    if not scores:
        return 0.0

    sequence = getattr(outputs, "sequences", None)
    if sequence is None or sequence.numel() == 0:
        return 0.0

    generated_steps = len(scores)
    token_ids = sequence[0][-generated_steps:]

    probabilities = []
    for logits, token_id_tensor in zip(scores, token_ids):
        token_id = int(token_id_tensor.item())
        vocab_size = logits.shape[-1]
        if token_id < 0 or token_id >= vocab_size:
            continue

        prob = torch.softmax(logits[0], dim=-1)[token_id].item()
        probabilities.append(prob)

    if not probabilities:
        return 0.0

    return float(sum(probabilities) / len(probabilities))


class BengaliRecognizer:
    def __init__(
        self,
        model_path: str = "models/recognizer/qween_vision_model_for_bangla",
        base_model_name: str = "unsloth/qwen3-vl-8b-instruct-bnb-4bit",
    ):
        logger.info("Initializing BengaliRecognizer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = self.device.type == "cuda"
        self.torch_dtype = torch.float16 if self.use_cuda else torch.float32
        logger.info(
            "BengaliRecognizer device setup | device=%s | dtype=%s",
            self.device,
            self.torch_dtype,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # unsloth/qwen3-vl-8b-instruct-bnb-4bit is already pre-quantized to 4-bit.
        # Do NOT pass quantization_config — the weights are already bnb-quantized.
        # bitsandbytes is still used under the hood to deserialize them.
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            # device_map="auto" if self.use_cuda else None,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            model_path,
            # device_map="auto" if self.use_cuda else None,
            # torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        # if not self.use_cuda:
        #     self.model.to(self.device)
        if self.use_cuda:
            self.model.to("cuda")
        else:
            self.model.to(self.device)    

        self.model.eval()
        logger.info("BengaliRecognizer initialized successfully")

    def inference(self, image: Image.Image) -> Tuple[str, float]:
        logger.info("BengaliRecognizer inference started")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Qwen VL requires chat-template formatting + process_vision_info
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image exactly as written. "
                            "Return only the extracted text, nothing else."
                        ),
                    },
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Strip the input prompt tokens from the output before decoding
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_len:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        confidence = _estimate_confidence(outputs)
        logger.info("BengaliRecognizer inference completed | confidence=%.2f", confidence)
        return text, round(confidence, 2)


class EnglishRecognizer:
    def __init__(self, model_path: str = "models/recognizer/english_trocr_model"):
        logger.info("Initializing EnglishRecognizer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("EnglishRecognizer initialized successfully | device=%s", self.device)

    def inference(self, image: Image.Image) -> Tuple[str, float]:
        logger.info("EnglishRecognizer inference started")
        if image.mode != "RGB":
            image = image.convert("RGB")

        pixel_values = (
            self.processor(images=image, return_tensors="pt")
            .pixel_values
            .to(self.device)
        )

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_new_tokens=256,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        confidence = _estimate_confidence(outputs)
        logger.info("EnglishRecognizer inference completed | confidence=%.2f", confidence)
        return text, round(confidence, 2)