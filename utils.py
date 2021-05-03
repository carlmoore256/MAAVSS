import json
import urllib
import urllib.request
import glob

def get_all_files(base_dir, ext):
    return glob.glob(f'{base_dir}/*/**/**.{ext}', recursive=True)

def save_json(out_path, data, indent=3):
  with open(out_path, 'w') as outfile:
    json.dump(data, outfile, sort_keys=False, indent=indent)
  print(f'wrote json to {out_path}')

def load_json(path):
  with open(path) as json_file:
      jfile = json.load(json_file)
  return jfile
  

def get_video_title(vid_id):
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % vid_id}
    query_string = urllib.parse.urlencode(params)
    url = "https://www.youtube.com/oembed"
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        return data['title']




