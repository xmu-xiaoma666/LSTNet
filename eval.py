import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
#from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
from models.SDATR import Transformer, TransformerEncoder, TransformerDecoderLayer, \
    ScaledDotProductAttention
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


import os
import json

def predict_captions(model, dataloader, text_field):
    import itertools
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores2(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Semantic-enhanced Dual Attention Transformer')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--features_path', type=str,default='/home/data/coco_grid_feats2.hdf5')
    parser.add_argument('--annotation_folder', type=str,default='/home/data/m2_annotations')
    args = parser.parse_args()

    print('Semantic-enhanced Dual Attention Transformer Evaluation')

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    dataset = COCO(image_field, text_field, 'coco/images/', args.annotation_folder, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    print('*'*50)
    print('test_dataset',len(test_dataset))
    text_field.vocab = pickle.load(open('vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                                     attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    
    data = torch.load('saved_models/Tra_best_test.pth')
    model.load_state_dict(data['state_dict'])

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, dict_dataloader_test, text_field)
    print(scores)
