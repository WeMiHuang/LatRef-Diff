import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from templates import *

#from templates_cls import *
from torchvision.utils import make_grid, save_image


domains=[#['Male_without','Male_with'],
         #['Male_with','Male_without'],
         #['Young_without','Young_with'],
         #['Young_with','Young_without'],
       ['Smile_without', 'Smile_with'],
         #['Smile_with', 'Smile_without']
         ]

tags=[2]         #Male:0,Young:1,Smile:2
t_attribute=[1]  #+:1,-:0
inputs = './imgs/10912.jpg'
ref = './imgs/18907.jpg'
guidance = 'latent'
use_mask = True



device = 'cuda:0'
conf = ffhq256_autoenc()
model = LitModel(conf)
state = torch.load('./checkpoints/ffhq256_autoenc/LatRef.ckpt', map_location='cpu')
model.load_state_dict(state['state_dict'], strict=False)
model.model.eval()
model.model.to(device)
model.extractors.eval()
model.extractors.to(device)
model.mappers.eval()
model.mappers.to(device)


transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

with torch.no_grad():
    for idx, domain in enumerate(domains):
        src_domain=domain[0]
        tar_domain=domain[1]
        task = '%s2%s' % (src_domain, tar_domain)

        path_fake = os.path.join('./test', task, guidance)
        os.makedirs(path_fake,exist_ok=True)

        x_src = transform(Image.open(inputs).convert('RGB')).unsqueeze(0).cuda()
        stochastic_x_T = model.encode_stochastic(x=x_src, x_start=x_src, fusion=model.extractors, tag=(tags[idx], 1-t_attribute[idx]),
                                                cond_s=x_src,
                                                use_ema=False)

        if guidance == 'latent':
            z = th.randn(stochastic_x_T.size(0), 4,32).cuda()
            prior = model.extractors.image(x_src)
            s_trg = model.mappers[tags[idx]](z, t_attribute[idx],prior)


        else:
            s_trg = transform(Image.open(ref).convert('RGB')).unsqueeze(0).cuda()


        if use_mask:
            file_name, _ = os.path.splitext(os.path.basename(inputs))
            file_name = file_name + '.png'
            mask_path = os.path.join('./imgs', file_name)
            mask = torch.from_numpy(np.array(Image.open(mask_path))).unsqueeze(0).cuda()

            gen = model.eval_sampler.sample_mask(model=model.model,
                                                 noise=stochastic_x_T,
                                                 tag=(tags[idx], t_attribute[idx]),
                                                 cond_s=s_trg,
                                                 fusion=model.extractors,
                                                 x_start=x_src, mask=mask)

        else:
            gen = model.eval_sampler.sample(model=model.model,
                                             noise=stochastic_x_T,
                                             tag=(tags[idx], t_attribute[idx]),
                                             cond_s=s_trg,
                                             fusion=model.extractors,
                                             x_start=x_src,inference=True)




        grid = (make_grid(gen.unsqueeze(0), nrow=1) + 1) / 2
        sample_dir = os.path.join(path_fake,f'{os.path.splitext(os.path.basename(inputs))[0]}.jpg')
        save_image(grid, sample_dir)

