import torch
import numpy as np

from networks import CLIPModel_full

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
    eval_model = CLIPModel_full(args)
    eval_model.eval() 
    
    
    texts = testloader.dataset.text 
    
    if args.dataset in ['flickr', 'coco']:
        if args.dataset == 'flickr':
            bert_test_embed = eval_model.text_encoder(texts)
        elif args.dataset == 'coco':
            bert_test_embed = torch.cat((eval_model.text_encoder(texts[:10000]), 
                                         eval_model.text_encoder(texts[10000:20000]), 
                                         eval_model.text_encoder(texts[20000:])), dim=0)
    else:
        raise NotImplementedError
    
    logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    print('Computing features for evaluation...')

    txt_embed = eval_model.text_projection(bert_test_embed.float().to(args.device)) 
    text_embeds = txt_embed / txt_embed.norm(dim=1, keepdim=True) #torch.Size([5000, 768])
    text_embeds = text_embeds.to(args.device)

    image_embeds = []
    for image, img_id in testloader: 
        image_feat = eval_model.image_encoder(image.to(args.device))
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
    score_matrix_i2t = torch.full((len(image_embeds),len(text_embeds)),-100.0).to(args.device) #torch.Size([1000, 5000])
    #for i, sims in enumerate(metric_logger.log_every(sims_matrix[0:sims_matrix.size(0) + 1], 50, header)): 
    for i, sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
        topk_sim, topk_idx = sims.topk(k=128, dim=0)
        score_matrix_i2t[i,topk_idx] = topk_sim #i:0-999, topk_idx:0-4999, find top k (k=128) similar text for each image
    
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(text_embeds),len(image_embeds)),-100.0).to(args.device)
    for i,sims in enumerate(sims_matrix[0:sims_matrix.size(0) + 1]): 
        topk_sim, topk_idx = sims.topk(k=128, dim=0)
        score_matrix_t2i[i,topk_idx] = topk_sim

    val_result = itm_eval(score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), 
                          testloader.dataset.txt2img, testloader.dataset.img2txt) 
    print("Img R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}\tTxt R@1: {}\tR@5: {}\tR@10: {}\tR@Mean: {}".format(
            val_result['img_r1'], val_result['img_r5'], val_result['img_r10'], val_result['img_r_mean'], 
            val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['txt_r_mean']))  
