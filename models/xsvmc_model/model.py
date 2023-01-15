from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageFont, ImageDraw
from joblib import load
from transformers import BertTokenizer
from keras.utils import pad_sequences
from xsvmlib.xsvmc import xSVMC

max_len = 4433
    
clf = load("models/xsvmc_model/xsvmc.joblib")
SVs = clf.support_vectors_

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

colormap = LinearSegmentedColormap.from_list('custom', 
                                       [(0, '#162cd9'),
                                        (1,   '#f2271f')], N=256)

def draw_text(draw_obj, text, pos_x, pos_y, prob, font):
  color = colormap(prob)[:3]
  draw_obj.text((pos_x, pos_y), text, fill=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255), font=font)

def create_img(l_text, prob, path, y_size):
  offset_y = 10
  offset_x = 10
  img_x = 800
  img_y = 600

  img = Image.new(mode="RGBA", size=(img_x, img_y), color = (255, 255, 255))
  txt = Image.new('RGBA', img.size, (255,255,255,0))
  draw = ImageDraw.Draw(txt)
  font = ImageFont.truetype("fonts\SpaceMono-Bold.ttf", 16)

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
  combined.save(path)

def render_text(text, misv, save_name):
  y_size = int(len(text) / 60 * 25)
  values = misv
  max_value = max_value(values)
  prob = (values / max_value)
  create_img(text, prob, "./%s.png" % save_name, y_size)

def preprocess_text(text):
  encoded_input = tokenizer(text, return_tensors='tf')
  tokens = encoded_input.input_ids[0]
  ready_tokens = pad_sequences(tokens, padding="post", maxlen=max_len)[0]
  text_tokens = tokenizer.convert_ids_to_tokens(ready_tokens)
  return ready_tokens, text_tokens, len(tokens)

def contextualized_prediction(text):
  tokens, text_tokens, original_len = preprocess_text(text)
  clean_text = text_tokens[1:original_len-1]
  topK = clf.predict_with_context(tokens)
  for i in range(len(topK)):
    pred = topK[i]
    mu_misv = SVs[pred.eval.mu_hat.misv_idx][1:original_len-1]
    nu_misv = SVs[pred.eval.nu_hat.misv_idx][1:original_len-1]
    render_text(clean_text, mu_misv, "%d-favor" % i)
    render_text(clean_text, nu_misv, "%d-contra" % i)