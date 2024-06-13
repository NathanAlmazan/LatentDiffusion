import os
import uuid
import torch
import numpy as np

from PIL import Image
from diffusion.vae import Decoder
from diffusion.unet import Diffusion
from diffusion.sampler import KLMSSampler
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)


@app.route('/label/generate', methods=['POST'])
def label_generate():
    try:
        body = request.json

        if body is None:
            return jsonify({'error': 'Invalid request body'}), 400

        labels = body['labels']

        if len(labels) == 0 or len(labels) > 12:
            return jsonify({'error': 'Label count should be between 0 to 12.'}), 400

        if len([label for label in labels if label < 0 or label > 500]) > 0:
            return jsonify({'error': 'Label does not exist.'}), 400

        # get label embeddings
        embeddings = np.load(os.path.join('transformer', 'checkpoint', 'embeddings.npy'), allow_pickle=True)
        embeddings = torch.from_numpy(embeddings[labels])
        embeddings = embeddings.float().cuda()

        # generate initial noise
        sampler = KLMSSampler(n_inference_steps=50)
        latents = torch.randn((len(labels), 4, 64, 64))
        latents *= sampler.initial_scale
        latents = latents.float().cuda()

        # denoise image using diffusion model
        diffusion = Diffusion().cuda()
        diffusion.load_state_dict(torch.load(os.path.join('diffusion', 'checkpoint', 'diffusion.pt')))
        diffusion.eval()

        with torch.autocast('cuda') and torch.inference_mode():
            for timestep in sampler.time_steps:
                input_latents = latents * sampler.get_input_scale()
                time_embedding = sampler.get_time_embedding(timestep).float().cuda()
                output = diffusion(input_latents, embeddings, time_embedding)

                latents = sampler.step(latents, output)

        # free memory
        diffusion.cpu()
        del diffusion
        torch.cuda.empty_cache()

        # decode latent space to image
        decoder = Decoder().cuda()
        decoder.load_state_dict(torch.load(os.path.join('diffusion', 'checkpoint', 'decoder.pt')))
        decoder.eval()

        with torch.autocast('cuda') and torch.inference_mode():
            images = decoder(latents)

        # free memory
        decoder.cpu()
        del decoder
        torch.cuda.empty_cache()

        # rescale image from -1 to 1 to 0 to 255
        images = images.detach().cpu()
        images = ((255.0 * images) / 2) + 127.5
        images = images.clamp(0, 255)
        # permute image to (batch, height, width, channel)
        images = images.permute(0, 2, 3, 1)
        # convert to numpy integer
        images = images.numpy().astype(np.uint8)

        # save images
        files = []
        for image in images:
            file = f"{uuid.uuid4()}.png"
            image = Image.fromarray(image)
            image.save(os.path.join('static', file))
            files.append(file)

        return jsonify({'image_urls': [f"https://imagine.automos.net/generated/{f}" for f in files]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generated/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
