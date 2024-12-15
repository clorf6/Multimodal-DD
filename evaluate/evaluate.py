import torch
import numpy as np
import argparse
import time
import sys
import os

from tqdm import tqdm


from networks import CLIPModel_full

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from dataset.utils import load_trainloader, load_testloader

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    print("TR: ", len(ranks))
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    print("IR: ", len(ranks))
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result



def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int,
                        help='train epochs')
    parser.add_argument('--train_batch_size', default=16, type=int,
                        help='train batch size')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--dataset', default='sync', type=str,
                        help='data prepare to train for eval model')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--dataset_root', default='/home/xun_ying/DD/results/coco/Fri-Dec-13-13-43-06-2024', type=str,
                        help='dataset dir')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of workers')
    parser.add_argument('--shuffle', default=True, type=bool,
                        help='shuffle the data')
    parser.add_argument('--drop_last', default=True, type=bool,
                        help='drop the last batch')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device')
    parser.add_argument('--distill', default=False, type=bool,
                        help='Whether the caption is needed to distill or not')
    parser.add_argument('--image_encoder', type=str, default='nfnet', 
                        choices=['clip', 'nfnet', 'vit', 'nf_resnet50'],  help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', 
                        choices=['bert', 'clip'], help='text encoder')
    parser.add_argument('--image_pretrained', type=bool, default=True, 
                        help='image_pretrained')
    parser.add_argument('--text_pretrained', type=bool, default=True, 
                        help='text_pretrained')
    parser.add_argument('--image_trainable', type=bool, default=True, 
                        help='image_trainable') 
    parser.add_argument('--text_trainable', type=bool, default=False, 
                        help='text_trainable')
    parser.add_argument('--has_image_projection', type=bool, default=False, help='None')
    args_train_eval_model = parser.parse_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='test batch size')
    parser.add_argument('--dataset', default='coco', type=str,
                        help='data prepare to test')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image size')
    parser.add_argument('--dataset_root', default='/home/xun_ying/DD/data/coco', type=str,
                        help='image root dir')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of workers')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device')
    args_eval = parser.parse_args()
    
    eval_model = CLIPModel_full(args_train_eval_model)
    eval_model = eval_model.to(args_train_eval_model.device)
    eval_model.train()
    
    optimizer_img = torch.optim.SGD(eval_model.image_encoder.parameters(), lr=args_train_eval_model.lr, momentum=0.9, weight_decay=0.0005)
    optimizer_txt = torch.optim.SGD(eval_model.text_projection.parameters(), lr=args_train_eval_model.lr, momentum=0.9, weight_decay=0.0005)
    trainloader = load_trainloader(args_train_eval_model)
    
    for epoch in range(args_train_eval_model.epochs):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        for i, data in enumerate(tqdm(trainloader)):
            image = data[0].to(args_train_eval_model.device)
            caption = data[1]
            n_b = image.shape[0]
            loss, acc = eval_model(image, caption)
            loss_avg += loss.item() * n_b
            acc_avg += acc
            num_exp += n_b
            optimizer_img.zero_grad()
            optimizer_txt.zero_grad()
            loss.backward()
            optimizer_img.step()
            optimizer_txt.step()
        print(f'Epoch{epoch}: loss_avg: {loss_avg / num_exp}, acc_avg: {acc_avg / num_exp}')  
    
    
    
    eval_model.eval() 
    
    with torch.no_grad():
        testloader = load_testloader(args_eval)
        texts = testloader.dataset.text 
        
        
        if args_eval.dataset in ['flickr', 'coco']:
            if args_eval.dataset == 'flickr':
                bert_test_embed = eval_model.text_encoder(texts)
            elif args_eval.dataset == 'coco':
                num = 20
                part_length = len(texts) // num
                remainder = len(texts) % num
                parts = []
                start = 0
                for i in range(num):
                    end = start + part_length + (1 if remainder > 0 else 0)
                    parts.append(texts[start: end])
                    start = end
                    remainder -= 1
                encoded_chunks = []
                for chunk in parts:
                    encoded_chunk = eval_model.text_encoder(chunk)
                    encoded_chunks.append(encoded_chunk)
                bert_test_embed = torch.cat(encoded_chunks, dim=0)
        else:
            raise NotImplementedError
        
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        print('Computing features for evaluation...')
        txt_embed = eval_model.text_projection(bert_test_embed.float().to(args_eval.device)) 
        text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True) 
        text_embeds = text_embeds.to(args_eval.device)
        
        image_embeds = []
        for image, img_id in tqdm(testloader): 
            image_feat = eval_model.image_encoder(image.to(args_eval.device))
            im_embed = image_feat / image_feat.norm(dim=1, keepdim=True)
            image_embeds.append(im_embed)
        image_embeds = torch.cat(image_embeds,dim=0)
        use_image_projection = False
        if use_image_projection:
            im_embed = eval_model.image_projection(image_embeds.float())
            image_embeds = im_embed / im_embed.norm(dim=1, keepdim=True)
        else:
            image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
            
        sims_matrix = logit_scale.exp() * image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(image_embeds),len(text_embeds)),-100.0).to(args_eval.device) #torch.Size([1000, 5000])
        #for i, sims in enumerate(metric_logger.log_every(sims_matrix[0:sims_matrix.size(0) + 1], 50, header)): 
        for i, sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
            topk_sim, topk_idx = sims.topk(k=128, dim=0)
            score_matrix_i2t[i,topk_idx] = topk_sim #i:0-999, topk_idx:0-4999, find top k (k=128) similar text for each image
        
        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(text_embeds),len(image_embeds)),-100.0).to(args_eval.device)
        for i,sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
            topk_sim, topk_idx = sims.topk(k=128, dim=0)
            score_matrix_t2i[i,topk_idx] = topk_sim

        val_result = itm_eval(score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy(), 
                            testloader.dataset.txt2img, testloader.dataset.img2txt) 
        print("Img R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}\tTxt R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}".format(
                val_result['img_r1'], val_result['img_r5'], val_result['img_r10'], val_result['img_r_mean'], 
                val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['txt_r_mean']))  

if __name__ == '__main__':
    evaluate()