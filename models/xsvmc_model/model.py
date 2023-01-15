from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFont, ImageDraw
from joblib import load
from transformers import BertTokenizer
from keras.utils import pad_sequences
from xsvmlib.xsvmc import xSVMC
import base64
from io import BytesIO

max_len = 4500
    
clf = load("models/xsvmc_model/xsvmc.joblib")
SVs = clf.support_vectors_

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

colormap = LinearSegmentedColormap.from_list('custom', 
                                       [(0, '#162cd9'),
                                        (1,   '#f2271f')], N=256)

def draw_text(draw_obj, text, pos_x, pos_y, prob, font):
  color = colormap(prob)[:3]
  draw_obj.text((pos_x, pos_y), text, fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255), font=font)

def create_img(l_text, prob, y_size):
  offset_y = 10
  offset_x = 10
  img_x = 800
  img_y = 600

  img = Image.new(mode="RGBA", size=(img_x, img_y), color = (255, 255, 255))
  txt = Image.new('RGBA', img.size, (255,255,255,0))
  draw = ImageDraw.Draw(txt)
  font = ImageFont.truetype("models/xsvmc_model/fonts/SpaceMono-Bold.ttf", 16)

  last_pos_x = 0
  combined = 0
  pos_y = 0
  for i in range(len(l_text)):
    palabra = l_text[i]
    if palabra.startswith("##"):
      palabra = palabra[2:]
      pos_x = (len(palabra)) * 10
    elif palabra not in [".", ","]:
      palabra = " " + palabra
      pos_x = (len(palabra)) * 10
      if last_pos_x + pos_x + offset_x >= img_x - 70:
        pos_y += 18
        last_pos_x = 0
    else:
      pos_x = (len(palabra)) * 10

    draw_text(draw, palabra, last_pos_x + offset_x, pos_y + offset_y, prob[i], font)
    last_pos_x += pos_x

  combined = Image.alpha_composite(img, txt)
  buffered = BytesIO()
  combined.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str

def render_text(text, misv):
  y_size = int(len(text) / 60 * 25)
  values = misv
  max_value = max(values)
  prob = (values / max_value)
  return create_img(text, prob, y_size)

def preprocess_text(text):
  encoded_input = tokenizer(text, return_tensors='tf')
  tokens = encoded_input.input_ids[0]
  ready_tokens = pad_sequences([tokens], padding="post", maxlen=max_len)[0]
  text_tokens = tokenizer.convert_ids_to_tokens(ready_tokens)
  return ready_tokens, text_tokens, len(tokens)

def contextualized_prediction(text):
  tokens, text_tokens, original_len = preprocess_text(text)
  clean_text = text_tokens[1:original_len-1]
  topK = clf.predict_with_context(tokens)
  response = []
  for i in range(len(topK)):
    pred = topK[i]
    mu_misv = SVs[pred.eval.mu_hat.misv_idx][1:original_len-1]
    nu_misv = SVs[pred.eval.nu_hat.misv_idx][1:original_len-1]
    b64_pro = render_text(clean_text, mu_misv)
    b64_contra = render_text(clean_text, nu_misv)
    response.append({
      'clase': pred.class_name,
      'favor': b64_pro,
      'contra': b64_contra
    })
  return response