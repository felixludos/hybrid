
### 70
#
#unbuffer python hybrid.py wae double 3dshapes
#
#unbuffer python hybrid.py wae double 3dshapes --model.lr 2e-3
#
#unbuffer python hybrid.py wae double 3dshapes --model.lr 5e-4
#
#unbuffer python hybrid.py wae double 3dshapes --model.lr 2e-4
#
#unbuffer python hybrid.py wae double 3dshapes --model.lr 1e-4
#
#
#unbuffer python hybrid.py vae double 3dshapes
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 4
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 0.25
#
#
#
#unbuffer python hybrid.py fwae double 3dshapes
#
#unbuffer python hybrid.py fvae double 3dshapes
#
#
#
#unbuffer python hybrid.py double 3dshapes
#
#
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.beta1 0
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr 1e-3 --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.lr 2e-4 --model.discriminator.lr 1e-3
#
#unbuffer python hybrid.py gan double 3dshapes --model.lr 2e-4 --model.discriminator.lr 2e-3
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr 8e-3
#
#unbuffer python hybrid.py gan double 3dshapes --model.optim_type rmsprop


### 71

#unbuffer python hybrid.py fvae double 3dshapes

### 76

#unbuffer python hybrid.py wae large-double celeba
#
#unbuffer python hybrid.py vae large-double celeba
#
#unbuffer python hybrid.py fwae large-double celeba
#
#unbuffer python hybrid.py fvae large-double celeba
#
#unbuffer python hybrid.py gan large-double celeba
#
#unbuffer python hybrid.py large-double celeba
#
#unbuffer python hybrid.py dwae large-double celeba
#
#unbuffer python hybrid.py dvae large-double celeba
#
#unbuffer python hybrid.py fdwae large-double celeba
#
#unbuffer python hybrid.py fdvae large-double celeba




### 77


#unbuffer python hybrid.py gan double 3dshapes
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.beta1 0 --model.lr 2e-4 --model.discriminator.lr 1e-3
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.optim_type rmsprop --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.optim_type rmsprop --model.lr 2e-4 --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.lr 2e-4 --model.discriminator.lr _x_ --model.disc_steps 4
#
#
#
#unbuffer python hybrid.py wae double 3dshapes
#
#
#unbuffer python hybrid.py wae double 3dshapes --model.nonlin prelu
#
#unbuffer python hybrid.py wae double 3dshapes --model.nonlin relu
#
#unbuffer python hybrid.py wae double 3dshapes --model.up_type nearest
#
#unbuffer python hybrid.py wae double 3dshapes --model.down_type conv
#
#unbuffer python hybrid.py wae double 3dshapes --model.down_type conv --model.up_type conv
#
#unbuffer python hybrid.py wae double 3dshapes --model.norm_type batch
#
#unbuffer python hybrid.py wae double 3dshapes --model.lr 5e-4
#
#unbuffer python hybrid.py wae double 3dshapes --model.beta1 0.5 --model.beta2 0.9
#
#unbuffer python hybrid.py wae double 3dshapes --model.beta1 0 --model.beta2 0.9
#
#
#unbuffer python hybrid.py wae double 3dshapes --model.weight_decay 1e-3
#
#unbuffer python hybrid.py wae double 3dshapes --model.weight_decay 1e-4
#
#unbuffer python hybrid.py wae double 3dshapes --model.weight_decay 1e-5
#
#unbuffer python hybrid.py wae double 3dshapes --model.weight_decay 1e-6
#
#
#
#unbuffer python hybrid.py vae double 3dshapes
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 4
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 0.25
#
#
#
#unbuffer python hybrid.py double 3dshapes
#
#unbuffer python hybrid.py double 3dshapes --model.gan_wt 1e-5
#
#unbuffer python hybrid.py double 3dshapes --model.gan_wt 1e-3
#
#
#unbuffer python hybrid.py double 3dshapes
#
#unbuffer python hybrid.py schenc double 3dshapes
#
#unbuffer python hybrid.py schenc double 3dshapes --model.gan_warm_start 20
#
#unbuffer python hybrid.py double 3dshapes --model.gan_warm_start 40
#
#unbuffer python hybrid.py double 3dshapes --model.gan_warm_start 20
#
#unbuffer python hybrid.py double 3dshapes --model.gan_warm_start 10
#
#
#unbuffer python hybrid.py fwae double 3dshapes
#
#unbuffer python hybrid.py fwae double 3dshapes --model.reg_prior 0.2
#
#unbuffer python hybrid.py fwae double 3dshapes --model.reg_prior 0.1
#
#unbuffer python hybrid.py fwae double 3dshapes --model.gan_wt 1e-4
#
#
#unbuffer python hybrid.py fvae double 3dshapes
#
#unbuffer python hybrid.py fvae double 3dshapes --model.reg_prior 0.2
#
#unbuffer python hybrid.py fvae double 3dshapes --model.reg_prior 0.1
#
#unbuffer python hybrid.py fvae double 3dshapes --model.gan_wt 1e-4
#
#
#
#unbuffer python hybrid.py dwae double 3dshapes
#
#unbuffer python hybrid.py dwae double 3dshapes --model.gan_wt 1e-3
#
#unbuffer python hybrid.py schenc dwae double 3dshapes
#
#unbuffer python hybrid.py dwae double 3dshapes --model.gan_warm_start 20
#
#
#unbuffer python hybrid.py dvae double 3dshapes
#
#
#
#unbuffer python hybrid.py fdwae double 3dshapes
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.gan_wt 1e-3
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.gan_warm_start 20
#
#
#unbuffer python hybrid.py fdvae double 3dshapes
#
#unbuffer python hybrid.py fdvae double 3dshapes --model.gan_wt 1e-3
#
#unbuffer python hybrid.py fdvae double 3dshapes --model.gan_warm_start 20


### 78

#unbuffer python hybrid.py wae double cr-3dshapes
#
#unbuffer python hybrid.py vae double cr-3dshapes
#
#unbuffer python hybrid.py fwae double cr-3dshapes
#
#unbuffer python hybrid.py fvae double cr-3dshapes
#
#unbuffer python hybrid.py gan double cr-3dshapes
#
#unbuffer python hybrid.py double cr-3dshapes
#
#unbuffer python hybrid.py dwae double cr-3dshapes
#
#unbuffer python hybrid.py dvae double cr-3dshapes
#
#unbuffer python hybrid.py fdwae double cr-3dshapes
#
#unbuffer python hybrid.py fdvae double cr-3dshapes


#unbuffer python hybrid.py gan double cr-3dshapes --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double cr-3dshapes --model.discriminator.lr _x_ --model.disc_steps 4 --model.lr 2e-4
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr _x_ --model.disc_steps 5
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr _x_ --model.disc_steps 4
#
#unbuffer python hybrid.py gan double 3dshapes --model.discriminator.lr _x_ --model.disc_steps 4


### 82

#unbuffer python hybrid.py wae double cr-3dshapes
#
#unbuffer python hybrid.py vae double cr-3dshapes
#
#unbuffer python hybrid.py fwae double cr-3dshapes
#
#unbuffer python hybrid.py fvae double cr-3dshapes
#
#unbuffer python hybrid.py gan double cr-3dshapes
#
#unbuffer python hybrid.py double cr-3dshapes
#
#unbuffer python hybrid.py dwae double cr-3dshapes
#
#unbuffer python hybrid.py dvae double cr-3dshapes
#
#unbuffer python hybrid.py fdwae double cr-3dshapes
#
#unbuffer python hybrid.py fdvae double cr-3dshapes
#
#unbuffer python hybrid.py fdwae double cr-3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py fdvae double cr-3dshapes --model.prob 0.5
#
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py fdvae double 3dshapes --model.prob 0.5


### 84

#unbuffer python hybrid.py wae large-double celeba
#
#unbuffer python hybrid.py vae large-double celeba
#
#unbuffer python hybrid.py fwae large-double celeba
#
#unbuffer python hybrid.py fvae large-double celeba
#
#unbuffer python hybrid.py gan large-double celeba
#
#unbuffer python hybrid.py large-double celeba
#
#unbuffer python hybrid.py dwae large-double celeba
#
#unbuffer python hybrid.py dvae large-double celeba
#
#unbuffer python hybrid.py fdwae large-double celeba
#
#unbuffer python hybrid.py fdvae large-double celeba
#
#unbuffer python hybrid.py fdwae large-double celeba --model.prob 0.5
#
#unbuffer python hybrid.py fdvae large-double celeba --model.prob 0.5


### 85

#unbuffer python hybrid.py gan large-double celeba
#
#
#unbuffer python hybrid.py gan double cr-3dshapes
#
#unbuffer python hybrid.py gan double 3dshapes
#
#unbuffer python hybrid.py gan double 3dshapes --model.disc_steps 8
#
#unbuffer python hybrid.py gan double 3dshapes --model.scheduler_type step --model.scheduler_step 80
#
#unbuffer python hybrid.py gan double 3dshapes --model.latent_dim 32
#
#
#unbuffer python hybrid.py gan double 3dshapes --model.up_type nearest
#
#unbuffer python hybrid.py gan double 3dshapes --model.down_type conv --model.up_type conv
#
#unbuffer python hybrid.py gan double 3dshapes --model.generator.down_type conv --model.generator.up_type conv
#
#unbuffer python hybrid.py gan double 3dshapes --model.generator.down_type conv --model.generator.up_type conv
#
#unbuffer python hybrid.py gan double 3dshapes --model.generator.down_type conv --model.generator.up_type conv --model.generator.squeeze True --model.generator.internal_channels [512,512,256,256,256]
#
#unbuffer python hybrid.py gan double 3dshapes --model.generator.squeeze True --model.generator.internal_channels [512,512,256,256,256]
#
#unbuffer python hybrid.py gan double 3dshapes --model.residual True


### 86

#unbuffer python hybrid.py wae double 3dshapes --model.generator.squeeze True --model.generator.internal_channels [512,512,256,256,256]
#
#unbuffer python hybrid.py wae double 3dshapes --model.squeeze True --model.internal_channels [256,256,256,512,512] --model.generator.internal_channels [512,512,256,256,256]


### 87

#unbuffer python hybrid.py dwae double 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py dvae double 3dshapes --model.prob 0.5


### 88

#unbuffer python hybrid.py vae double 3dshapes  --model.up_type nearest
#
#unbuffer python hybrid.py fwae double 3dshapes  --model.up_type nearest
#
#unbuffer python hybrid.py fvae double 3dshapes --model.up_type nearest
#
#unbuffer python hybrid.py double 3dshapes --model.up_type nearest
#
#unbuffer python hybrid.py dwae double 3dshapes --model.up_type nearest
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.up_type nearest

### 89

#unbuffer python hybrid.py vae double 3dshapes  --model.up_type nearest --model.latent_reg_wt 2
#
#unbuffer python hybrid.py vae double 3dshapes  --model.up_type nearest --model.latent_reg_wt 4
#
#unbuffer python hybrid.py vae double 3dshapes  --model.up_type nearest --model.latent_reg_wt 8


### 92


#unbuffer python hybrid.py vae adain-dec 3dshapes
#
#unbuffer python hybrid.py fwae adain-dec 3dshapes
#
#unbuffer python hybrid.py fvae adain-dec 3dshapes
#
#unbuffer python hybrid.py adain-dec 3dshapes
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes

#unbuffer python hybrid.py gan adain-dec 3dshapes


### 94

#unbuffer python hybrid.py vae double 3dshapes
#
#unbuffer python hybrid.py fwae double 3dshapes
#
#unbuffer python hybrid.py fvae double 3dshapes
#
#unbuffer python hybrid.py double 3dshapes
#
#unbuffer python hybrid.py dwae double 3dshapes
#
#unbuffer python hybrid.py fdwae double 3dshapes
#
#unbuffer python hybrid.py gan double 3dshapes



### 95

#unbuffer python hybrid.py vae double 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py fwae double 3dshapes  --model.latent_dim 8
#
#unbuffer python hybrid.py fvae double 3dshapes  --model.latent_dim 8
#
#unbuffer python hybrid.py double 3dshapes  --model.latent_dim 8
#
#unbuffer python hybrid.py dwae double 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py gan double 3dshapes --model.latent_dim 8


### 96

#unbuffer python hybrid.py vae double 3dshapes --model.latent_dim 32
#
#unbuffer python hybrid.py fwae double 3dshapes  --model.latent_dim 32
#
#unbuffer python hybrid.py fvae double 3dshapes  --model.latent_dim 32
#
#unbuffer python hybrid.py double 3dshapes  --model.latent_dim 32
#
#unbuffer python hybrid.py dwae double 3dshapes --model.latent_dim 32
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 32
#
#unbuffer python hybrid.py gan double 3dshapes --model.latent_dim 32

### 97

#unbuffer python hybrid.py vae adain-dec 3dshapes
#
#unbuffer python hybrid.py fwae adain-dec 3dshapes
#
#unbuffer python hybrid.py fvae adain-dec 3dshapes
#
#unbuffer python hybrid.py adain-dec 3dshapes
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes
#
#unbuffer python hybrid.py gan adain-dec 3dshapes

### 98

#unbuffer python hybrid.py double 3dshapes  --model.latent_dim 32

### 99

#unbuffer python hybrid.py wae adain-dec 3dshapes

### 100

#unbuffer python hybrid.py wae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py vae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py fwae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py fvae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py gan adain-dec 3dshapes --model.latent_dim 8

### 101

#unbuffer python hybrid.py dwae adain-dec 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes --model.prob 0.5

### 102

#unbuffer python hybrid.py wae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py vae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py fwae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py fvae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py dwae adain-dec 3dshapes --model.prob 0.5 --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py dvae adain-dec 3dshapes --model.prob 0.5 --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py fdwae adain-dec 3dshapes --model.prob 0.5 --model.generator.adain_latent_dim 16 --model.generator.splits 4
#
#unbuffer python hybrid.py gan adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4

### 103

#unbuffer python hybrid.py gan adain-dec 3dshapes --model.generator.adain_latent_dim 16 --model.generator.splits 4

### 104

#unbuffer python hybrid.py wae adain-dec 3dshapes

### 105

#unbuffer python hybrid.py fvae adain-dec 3dshapes --model.latent_dim 8

### 106

#unbuffer python hybrid.py fvae adain-dec 3dshapes --model.latent_dim 8

### 107

#unbuffer python hybrid.py --resume 3dshapes-fwae_0102-5966133-02_200109-133116


### 108

#unbuffer python hybrid.py --resume 3dshapes-dvae_0101-5966132-01_200109-133045
#
#unbuffer python hybrid.py --resume 3dshapes-dvae_0101-5966132-02_200109-133053

### 109-118

#unbuffer python hybrid.py --resume 3dshapes-fwae_0102-5966133-02_200109-133116


### 119-120

#unbuffer python hybrid.py --resume 3dshapes-dvae_0101-5966132-02_200109-133053

### 121

#unbuffer python hybrid.py --resume 3dshapes-dvae_0101-5966132-01_200109-133045
#
#unbuffer python hybrid.py --resume 3dshapes-dvae_0101-5966132-02_200109-133053


### 122

#unbuffer python hybrid.py dwae double 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py dvae double 3dshapes
#
#unbuffer python hybrid.py dvae double 3dshapes --model.prob 0.5
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.prob 0.5

### 123

#unbuffer python hybrid.py --resume 3dshapes-fvae_0102-5966133-03_200109-133107
#
#unbuffer python hybrid.py --resume 3dshapes-vae_0102-5966133-01_200109-133106
#
#unbuffer python hybrid.py --resume 3dshapes-wae_0102-5966133-00_200109-133106

### 124

#unbuffer python hybrid.py --resume 3dshapes-dvae_0100-5966131-07_200109-133026
#unbuffer python hybrid.py --resume 3dshapes-dvae_0100-5966131-08_200109-133040
#unbuffer python hybrid.py --resume 3dshapes-dwae_0100-5966131-05_200109-133027
#unbuffer python hybrid.py --resume 3dshapes-dwae_0100-5966131-06_200109-133124
#unbuffer python hybrid.py --resume 3dshapes-fvae_0100-5966131-03_200109-133036
#unbuffer python hybrid.py --resume 3dshapes-vae_0100-5966131-01_200109-133046

### 125

#unbuffer python hybrid.py --resume 3dshapes-dwae_0097-5966051-04_200108-181233
#unbuffer python hybrid.py --resume 3dshapes-fdwae_0097-5966051-05_200108-181229
#unbuffer python hybrid.py --resume 3dshapes-wgan_0097-5966051-06_200108-181229
#unbuffer python hybrid.py --resume 3dshapes-wpp_0097-5966051-03_200108-181238

### 126

#unbuffer python hybrid.py --resume 3dshapes-fvae_0094-5966044-02_200108-175515
#unbuffer python hybrid.py --resume 3dshapes-wgan_0094-5966044-06_200108-175507
#unbuffer python hybrid.py --resume 3dshapes-wpp_0094-5966044-03_200108-175513

### 127

#unbuffer python hybrid.py --resume 3dshapes-wgan_0095-5966045-06_200108-175835
#unbuffer python hybrid.py --resume 3dshapes-wpp_0095-5966045-03_200108-175832
#unbuffer python hybrid.py --resume 3dshapes-dwae_0095-5966045-04_200108-175850

### 128

#unbuffer python hybrid.py --resume 3dshapes-fwae_0096-5966047-01_200108-175918


### 129

#unbuffer python hybrid.py --resume 3dshapes-vae_0102-5966133-01_200109-133106


#### 130

#unbuffer python hybrid.py dwae double 3dshapes --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py dvae double 3dshapes --model.latent_dim 8
#
#unbuffer python hybrid.py dvae double 3dshapes --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.prob 0.5 --model.latent_dim 8


### 131

#unbuffer python hybrid.py dwae double 3dshapes --model.prob 0.5 --model.latent_dim 32
#
#unbuffer python hybrid.py dvae double 3dshapes --model.latent_dim 32
#
#unbuffer python hybrid.py dvae double 3dshapes --model.prob 0.5 --model.latent_dim 32
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.prob 0.5 --model.latent_dim 32


### 132

#unbuffer python hybrid.py wae double redball
#
#unbuffer python hybrid.py vae double redball
#
#unbuffer python hybrid.py fwae double redball
#
#unbuffer python hybrid.py fvae double redball
#
#unbuffer python hybrid.py double redball
#
#unbuffer python hybrid.py dwae double redball --model.prob 0.5
#
#unbuffer python hybrid.py dvae double redball --model.prob 0.5
#
#unbuffer python hybrid.py fdwae double redball
#
#unbuffer python hybrid.py fdwae double redball --model.prob 0.5
#
#unbuffer python hybrid.py gan double redball


#OPENAI_LOGDIR=/is/ei/fleeb/workspace/clones/baselines_tf1/logs/cartpole-ppo OPENAI_LOG_FORMAT=csv python -m baselines.run --alg=ppo2 --env=CartPole-v0 --num_timesteps=30000 --nsteps=128 --save_path=/is/ei/fleeb/workspace/clones/baselines_tf1/logs/cartpole-ppo

### 133-136

#unbuffer python hybrid.py wae double rgbball
#
#unbuffer python hybrid.py vae double rgbball
#
#unbuffer python hybrid.py fwae double rgbball
#
#unbuffer python hybrid.py fvae double rgbball
#
#unbuffer python hybrid.py double rgbball
#
#unbuffer python hybrid.py dwae double rgbball --model.prob 0.5
#
#unbuffer python hybrid.py dvae double rgbball --model.prob 0.5
#
#unbuffer python hybrid.py fdwae double rgbball
#
#unbuffer python hybrid.py fdwae double rgbball --model.prob 0.5
#
#unbuffer python hybrid.py gan double rgbball
#
#unbuffer python hybrid.py fvae double rgbball --model.latent_dim 8
#
#unbuffer python hybrid.py fvae adain-dec rgbball --model.latent_dim 8
#
#unbuffer python hybrid.py fdvae double rgbball --model.latent_dim 8 --model.prob 0.5
#
#unbuffer python hybrid.py fdvae adain-dec rgbball --model.latent_dim 8 --model.prob 0.5


### 137

#unbuffer python hybrid.py fvae double 3dshapes


### 138

#unbuffer python hybrid.py wae double 3dshapes
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 2
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 4
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 8
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 16
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 64


### 144


#unbuffer python hybrid.py wae large-double atari --game asterix
#
#unbuffer python hybrid.py vae large-double atari --game asterix
#
#unbuffer python hybrid.py fvae large-double atari --game asterix
#
#unbuffer python hybrid.py fdwae large-double atari --game asterix --model.prob 0.5


### 145

#unbuffer python hybrid.py wae double atari --game pacman
#
#unbuffer python hybrid.py wae large-double atari --game pacman
#
#unbuffer python hybrid.py vae large-double atari --game pacman
#
#unbuffer python hybrid.py fvae large-double atari --game pacman
#
#unbuffer python hybrid.py fdwae large-double atari --game pacman --model.prob 0.5
#
#unbuffer python hybrid.py gan large-double atari --game pacman


### 146

#echo "works once"
#
#echo "works again"

### 147-156

#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --save-dir /home/fleeb/rl/MsPacman/$PROCESS_ID --seed $PROCESS_ID


### 157

#unbuffer python hybrid.py dwae double 3dshapes --model.latent_reg_wt 0
#
#unbuffer python hybrid.py fwae double 3dshapes --model.reg_prior 0

#unbuffer python hybrid.py fdwae double 3dshapes --model.reg_prior 0



### 158

#unbuffer python hybrid.py fdwae double 3dshapes --model.reg_prior 0


### 159

#unbuffer python hybrid.py --resume 3dshapes-fvae_0137-5985626-00_200116-114105
#
#unbuffer python hybrid.py --resume 3dshapes-vae_0138-5986313-05_200116-114439



### 160


#unbuffer python hybrid.py wae double atari --game asterix --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py vae double atari --game asterix --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64
#
#unbuffer python hybrid.py fvae double atari --game asterix --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64
#
#unbuffer python hybrid.py fdwae double atari --game asterix --model.prob 0.5 --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64


### 161

#unbuffer python hybrid.py wae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py wae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py vae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py fvae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py fdwae double atari --game pacman --model.prob 0.5 --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py gan double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64


### 162

#unbuffer python hybrid.py wae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py vae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py fvae double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py fdwae double atari --game pacman --model.prob 0.5 --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py gan double atari --game pacman --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#
#unbuffer python hybrid.py wae double atari --game asterix --training.epochs 60 --scheduler_patience 4 --dataset.batch_size 64
#
#unbuffer python hybrid.py vae double atari --game asterix --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64
#
#unbuffer python hybrid.py fvae double atari --game asterix --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64
#
#unbuffer python hybrid.py fdwae double atari --game asterix --model.prob 0.5 --training.epochs 60 --scheduler_patience 4  --dataset.batch_size 64


### 163

#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/10 --save-dir /home/fleeb/rl/MsPacman/10 --seed 10
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/12 --save-dir /home/fleeb/rl/MsPacman/12 --seed 12
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/13 --save-dir /home/fleeb/rl/MsPacman/13 --seed 13
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/14 --save-dir /home/fleeb/rl/MsPacman/14 --seed 14
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/15 --save-dir /home/fleeb/rl/MsPacman/15 --seed 15
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/16 --save-dir /home/fleeb/rl/MsPacman/16 --seed 16
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/17 --save-dir /home/fleeb/rl/MsPacman/17 --seed 17
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/18 --save-dir /home/fleeb/rl/MsPacman/18 --seed 18
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/19 --save-dir /home/fleeb/rl/MsPacman/19 --seed 19
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/5 --save-dir /home/fleeb/rl/MsPacman/5 --seed 5
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/7 --save-dir /home/fleeb/rl/MsPacman/7 --seed 7
#unbuffer python main.py --env-name "MsPacmanNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/MsPacman/9 --save-dir /home/fleeb/rl/MsPacman/9 --seed 9

### 164

#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "AsterixNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Asterix/$PROCESS_ID --save-dir /home/fleeb/rl/Asterix/$PROCESS_ID --seed $PROCESS_ID


### 165

#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "SeaquestNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --save-dir /home/fleeb/rl/Seaquest/$PROCESS_ID --seed $PROCESS_ID


### 166

#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID
#unbuffer python main.py --env-name "BreakoutNoFrameskip-v4" --algo acktr --num-processes 32 --num-steps 20 --log-dir /home/fleeb/rl/Breakout/$PROCESS_ID --save-dir /home/fleeb/rl/Breakout/$PROCESS_ID --seed $PROCESS_ID


### 167

#unbuffer python hybrid.py dwae double 3dshapes --model.latent_reg_wt 0 --model.prob 0.5
#
#unbuffer python hybrid.py fwae double 3dshapes --model.reg_prior 0 --model.prob 0.5
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.reg_prior 0 --model.prob 0.5


### 168

#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.25 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e4 --model.reg_prior 0.25 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.1 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e4 --model.reg_prior 0.1 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.05 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e4 --model.reg_prior 0.05 --training.epochs 60

### 169

#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 500 --model.reg_prior 0.05 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 500 --model.reg_prior 0.01 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.01 --training.epochs 60
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e4 --model.reg_prior 0.01 --training.epochs 60
#
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 500 --model.reg_prior 0.05 --training.epochs 60 --model.latent_dim 8
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 500 --model.reg_prior 0.01 --training.epochs 60 --model.latent_dim 8
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.01 --training.epochs 60 --model.latent_dim 8
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e4 --model.reg_prior 0.01 --training.epochs 60 --model.latent_dim 8


### 170

#unbuffer python hybrid.py fwae double 3dshapes --model.latent_reg_wt 1e3 --model.reg_prior 0.05 --training.epochs 60 --model.latent_dim 8
#
#
#unbuffer python hybrid.py dwae double 3dshapes --model.latent_reg_wt 0.01 --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py dvae double 3dshapes --model.reg_prior 0.01 --model.prob 0.5 --model.latent_dim 8
#
#unbuffer python hybrid.py vae double 3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 1e3 --model.latent_dim 8
#
#unbuffer python hybrid.py fvae double 3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 1e3 --model.latent_dim 8

### 171

#unbuffer python hybrid.py vae double 3dshapes --model.latent_reg_wt 0.01 --model.latent_dim 8
#

### 172

#unbuffer python hybrid.py wae double 3dshapes --model.latent_reg_wt 0 --model.latent_dim 8


### 173

#unbuffer python hybrid.py --resume 3dshapes-wae_0172-6005496-00_200122-163522


### 174

#unbuffer python hybrid.py wae double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py wae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 0
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py vae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py fwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.01
#
#
#unbuffer python hybrid.py fvae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 2
#
#unbuffer python hybrid.py fvae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 8
#
#unbuffer python hybrid.py fvae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 5 --model.reg_prior 0.2
#
#
#unbuffer python hybrid.py double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py dwae double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py gan double 3dshapes --model.latent_dim 12


### 175

#unbuffer python hybrid.py wae double 3dshapes --model.latent_dim 12


### 176

#unbuffer python hybrid.py wae double 3dshapes --model.latent_dim 12 --model.generator.adain_latent_dim 8 --model.generator.splits 2


### 177


#unbuffer python hybrid.py wae double noisy-3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py wae double nocap-3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py wae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 0
#
#
#unbuffer python hybrid.py vae double nocap-3dshapes --model.latent_dim 12
#
#unbuffer python hybrid.py vae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py fwae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py fwae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#
#unbuffer python hybrid.py fvae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 2
#
#unbuffer python hybrid.py fvae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 5 --model.reg_prior 0.
#
#
#unbuffer python hybrid.py dwae double nocap-3dshapes --model.latent_dim 12
#
#
#unbuffer python hybrid.py fdwae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 0
#
#unbuffer python hybrid.py fdwae double nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1


### 178

#unbuffer python hybrid.py wae double transfer/cyl --model.latent_dim 12
#
#unbuffer python hybrid.py wae double transfer/cyl --model.latent_dim 12
#
#unbuffer python hybrid.py wae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 0
#
#
#unbuffer python hybrid.py vae double transfer/cyl --model.latent_dim 12
#
#unbuffer python hybrid.py vae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py fwae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py fwae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#
#unbuffer python hybrid.py fvae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 2
#
#unbuffer python hybrid.py fvae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 5 --model.reg_prior 0.

### 179

#unbuffer python hybrid.py fdwae double transfer/cyl --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py dwae double transfer/cyl --model.latent_dim 12


### 180

#unbuffer python hybrid.py transfer/cyl-ball --load cyl-wae_0178-6017981-00_200125-221552
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-wae_0178-6017981-02_200125-222200
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-vae_0178-6017981-03_200125-222128
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-vae_0178-6017981-04_200125-222245
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-fwae_0178-6017981-05_200125-222334
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-fwae_0178-6017981-06_200125-222430
#unbuffer python hybrid.py transfer/cyl-ball --load cyl-fvae_0178-6017981-07_200125-222631

### 181



#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 0 --model.prob 0.1
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.prob 0.1
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 0 --model.gan_wt 0.00005
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.gan_wt 0.00005
#
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 0 --model.gan_wt 0.00001
#
#unbuffer python hybrid.py fdwae double 3dshapes --model.latent_dim 12 --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.gan_wt 0.00001





###

#unbuffer python hybrid.py vae adain-dec nocap-3dshapes --model.latent_dim 12 --model.latent_reg_wt 4

#unbuffer python hybrid.py fvae double nocap-3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 10 --model.latent_dim 8
#
#unbuffer python hybrid.py fvae double nocap-3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 100 --model.latent_dim 8
#
#unbuffer python hybrid.py fvae double nocap-3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 1e3 --model.latent_dim 8
#
#
#unbuffer python hybrid.py fdwae double nocap-3dshapes --model.reg_prior 0.01 --model.latent_reg_wt 1e3 --model.prob 0.5 --model.latent_dim 8

















































####################################################################3

### 0

#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.prob 0.1
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 0 --model.prob 0.1
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.prob 0.1
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 0 --model.gan_wt 0.00005
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.gan_wt 0.00005
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 0 --model.gan_wt 0.00001
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.gan_wt 0.00001

### 1

#unbuffer python hybrid.py n/gan n/double n/3dshapes
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes

### 2

#unbuffer python hybrid.py n/wae n/double n/3dshapes
#
#unbuffer python hybrid.py n/wae n/double n/3dshapes --model.latent_reg_wt 10000
#
#unbuffer python hybrid.py n/wae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#
#unbuffer python hybrid.py n/fvae n/double n/3dshapes --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/fvae n/double n/3dshapes --model.latent_reg_wt 5 --model.reg_prior 0.2
#
#
#unbuffer python hybrid.py n/wae n/double n/t/box
#
#unbuffer python hybrid.py n/vae n/double n/t/box --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/fwae n/double n/t/box --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fvae n/double n/t/box --model.latent_reg_wt 2

### 3

#unbuffer python hybrid.py n/gan n/double n/3dshapes
#
#unbuffer python hybrid.py n/wae n/double n/3dshapes
#

### 4

#unbuffer python hybrid.py n/fdwae n/double n/3dshapes --model.prob 0.2
#
#unbuffer python hybrid.py n/fdwae n/double n/3dshapes
#
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.prob 0.1
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.prob 0.2
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes
#
#
#unbuffer python hybrid.py n/dvae n/double n/3dshapes --model.prob 0.1
#
#unbuffer python hybrid.py n/dvae n/double n/3dshapes --model.prob 0.2
#
#unbuffer python hybrid.py n/dvae n/double n/3dshapes

### 5-6

#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-fvae_0002-6026518-13_200129-002149 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-fwae_0002-6026518-12_200129-002149 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-vae_0002-6026518-11_200129-002213 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-wae_0002-6026518-10_200129-001725 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-ae_0002-6026518-03_200129-001602 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dvae_0004-6026551-05_200129-003640 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dvae_0004-6026551-06_200129-003638 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dvae_0004-6026551-07_200129-003721 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dwae_0004-6026551-02_200129-002921 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dwae_0004-6026551-03_200129-003324 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dwae_0004-6026551-04_200129-003405 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-01_200129-001059 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-02_200129-000936 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-03_200129-001058 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-04_200129-001059 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-05_200129-000937 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-06_200129-001100 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0000-6026515-07_200129-001057 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0004-6026551-00_200129-003007 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fdwae_0004-6026551-01_200129-002926 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fvae_0002-6026518-08_200129-001754 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fvae_0002-6026518-09_200129-001837 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fwae_0002-6026518-06_200129-001738 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fwae_0002-6026518-07_200129-001835 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-vae_0002-6026518-04_200129-001602 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-vae_0002-6026518-05_200129-001620 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-wae_0002-6026518-01_200129-001522 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-wae_0002-6026518-02_200129-001602 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-wae_0003-6026539-01_200129-002624 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-wgan_0001-6026517-02_200129-000957 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-wgan_0003-6026539-00_200129-002531 --extend 150000


### 7


#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes
#
#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.latent_reg_wt 10000
#
#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 5 --model.reg_prior 0.2
#
#
#unbuffer python hybrid.py n/dvae n/adain-dec n/3dshapes
#
#unbuffer python hybrid.py n/dwae n/adain-dec n/3dshapes
#
#unbuffer python hybrid.py n/fdwae n/adain-dec n/3dshapes
#
#
#unbuffer python hybrid.py n/wae n/adain-dec n/t/box
#
#unbuffer python hybrid.py n/vae n/adain-dec n/t/box --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/t/box --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/t/box --model.latent_reg_wt 2


### 8

#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4


### 9

#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.latent_reg_wt 4 --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 2 --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/dvae n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/dwae n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/fdwae n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#
#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.latent_reg_wt 4 --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 2 --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/dvae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/dwae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/fdwae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4


### 10

#unbuffer python hybrid.py n/dvae n/double n/t/box
#
#unbuffer python hybrid.py n/dwae n/double n/t/box
#
#unbuffer python hybrid.py n/fdwae n/double n/t/box


### 11

#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-wae_0007-6027503-12_200129-124338 --extend 80000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-fwae_0007-6027503-14_200129-124340 --extend 80000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-fvae_0007-6027503-15_200129-124413 --extend 80000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-vae_0007-6027503-13_200129-124415 --extend 80000

### 12

#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.prob 0.2
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.prob 0.5
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.prob 0.8
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.prob 1

### 13

#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.prob 0.1
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.gan_wt 0.001
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.gan_wt 0.0005
#
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.gan_wt 0.00005
#unbuffer python hybrid.py n/dwae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type dae --model.gan_wt 0.00001

### 14

#unbuffer python hybrid.py n/wae n/large-double n/celeba --model.latent_reg_wt 0 --info.model_type ae
#
#unbuffer python hybrid.py n/wae n/large-double n/celeba
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/large-double n/celeba --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/large-double n/celeba --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/large-double n/celeba --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/large-double n/celeba
#
#unbuffer python hybrid.py n/dwae n/large-double n/celeba  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/celeba --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/celeba --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/wae n/large-adain n/celeba --model.latent_reg_wt 0 --info.model_type ae
#
#unbuffer python hybrid.py n/wae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/vae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/vae n/large-adain n/celeba --model.latent_reg_wt 4

### 15

#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-dvae_0010-6027546-00_200129-125909 --extend 80000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-dwae_0010-6027546-01_200129-130102 --extend 80000
#unbuffer python hybrid.py --legacy 0 --resume 3dSshape1-fdwae_0010-6027546-02_200129-130128 --extend 80000

### 16

#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-wae_0002-6026518-10_200129-001725/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-vae_0002-6026518-11_200129-002213/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-fwae_0002-6026518-12_200129-002149/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-fvae_0002-6026518-13_200129-002149/checkpoint_27.pth.tar
#



### 17

#unbuffer python hybrid.py n/wae n/double n/t/box-s1-ctrl
#unbuffer python hybrid.py n/vae n/double n/t/box-s1-ctrl --model.latent_reg_wt 4
#unbuffer python hybrid.py n/fwae n/double n/t/box-s1-ctrl --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fvae n/double n/t/box-s1-ctrl --model.latent_reg_wt 2

### 18

#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type batch
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type None
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type group
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.lr 2.e-3
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.up_type bilinear


### 19


#unbuffer python hybrid.py n/gan n/large-double n/celeba
#
#unbuffer python hybrid.py n/gan n/large-adain n/celeba
#
#unbuffer python hybrid.py n/gan n/large-adain n/celeba --model.generator.splits 8 --model.generator.adain_latent_dim 40 --model.latent_dim 40


### 20-21

#unbuffer python hybrid.py n/wae n/double n/t/box-s1 --training.step_limit 100000
#unbuffer python hybrid.py n/vae n/double n/t/box-s1 --model.latent_reg_wt 4 --training.step_limit 100000
#unbuffer python hybrid.py n/fwae n/double n/t/box-s1 --model.latent_reg_wt 1000 --model.reg_prior 0.1 --training.step_limit 100000
#unbuffer python hybrid.py n/fvae n/double n/t/box-s1 --model.latent_reg_wt 2 --training.step_limit 100000


### 22


#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-01_200129-181926 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-02_200129-181947 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-03_200129-181951 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-04_200129-181936 --extend 150000


### 23

#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-wae_0007-6027503-12_200129-124338/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-vae_0007-6027503-13_200129-124415/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-fwae_0007-6027503-14_200129-124340/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load 3dSshape1-fvae_0007-6027503-15_200129-124413/checkpoint_27.pth.tar


### 24

#unbuffer python hybrid.py n/t/box-s2 --load 3dSshape1-wae_0002-6026518-10_200129-001725/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s2 --load 3dSshape1-vae_0002-6026518-11_200129-002213/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s2 --load 3dSshape1-fwae_0002-6026518-12_200129-002149/checkpoint_27.pth.tar
#unbuffer python hybrid.py n/t/box-s2 --load 3dSshape1-fvae_0002-6026518-13_200129-002149/checkpoint_27.pth.tar


### 25

#unbuffer python hybrid.py n/wae n/adain-dec n/t/box-s1
#unbuffer python hybrid.py n/vae n/adain-dec n/t/box-s1 --model.latent_reg_wt 4
#unbuffer python hybrid.py n/fwae n/adain-dec n/t/box-s1 --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fvae n/adain-dec n/t/box-s1 --model.latent_reg_wt 2

### 26

#unbuffer python hybrid.py n/wae n/large-double n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#
#unbuffer python hybrid.py n/wae n/large-double n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-double n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-double n/3dshapes --model.latent_reg_wt 4
#
#
#unbuffer python hybrid.py n/wae n/large-adain n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#
#unbuffer python hybrid.py n/wae n/large-adain n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-adain n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-adain n/3dshapes --model.latent_reg_wt 4

### 27


#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-01_200129-181926 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-02_200129-181947 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-03_200129-181951 --extend 150000
#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-dae_0012-6032710-04_200129-181936 --extend 150000


### 28

#unbuffer python hybrid.py n/ae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/wae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.latent_reg_wt 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 2 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/double n/3dshapes --model.norm_type group --model.up_type bilinear


### 29

#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.latent_reg_wt 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fvae n/double n/3dshapes --model.latent_reg_wt 2 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear

### 30

#unbuffer python hybrid.py n/ae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#
#unbuffer python hybrid.py n/dae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear


### 31

#unbuffer python hybrid.py n/fae n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear


### 32


#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear


### 33

#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear

### 34

#unbuffer python hybrid.py --legacy 0 --load 3dshapes-vae_0029-6046947-02_200131-125033/checkpoint_3.pth.tar


### 35


#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/wae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/vae n/adain-dec n/3dshapes --model.latent_reg_wt 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/dae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/3dshapes --model.latent_reg_wt 2 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/fae n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear


### 36

#unbuffer python hybrid.py n/gan n/double n/3dshapes --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.norm_type group --model.up_type bilinear

### 37

#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fvae_0035-6048636-08_200131-185352

### 38

#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-fae_0031-6046954-00_200131-132552

### 39-42

#unbuffer python main_eval.py --names 3dshapes-ae_0030-6046949-00_200131-125923 --save-dir /home/fleeb/results/test --overwrite

### 43

#unbuffer python main_eval.py --names 3dshapes-wae_0028-6046939-01_200131-124832 --save-dir /home/fleeb/results/test --overwrite
#unbuffer python main_eval.py --names 3dshapes-vae_0028-6046939-02_200131-124807 --save-dir /home/fleeb/results/test --overwrite
#unbuffer python main_eval.py --names 3dshapes-vae_0028-6046939-03_200131-124821 --save-dir /home/fleeb/results/test --overwrite
#unbuffer python main_eval.py --names 3dshapes-fwae_0028-6046939-05_200131-124815 --save-dir /home/fleeb/results/test --overwrite
#unbuffer python main_eval.py --names 3dshapes-fvae_0029-6046947-06_200131-125026 --save-dir /home/fleeb/results/test --overwrite


### 45


#unbuffer python hybrid.py n/ae n/double n/t/box
#
#unbuffer python hybrid.py n/wae n/double n/t/box
#
#unbuffer python hybrid.py n/vae n/double n/t/box
#
#unbuffer python hybrid.py n/vae n/double n/t/box --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/fae n/double n/t/box
#
#unbuffer python hybrid.py n/dae n/double n/t/box
#
#unbuffer python hybrid.py n/fwae n/double n/t/box --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fvae n/double n/t/box --model.latent_reg_wt 2


### 46

#unbuffer python hybrid.py n/ae n/double n/t/nocap

#unbuffer python hybrid.py n/wae n/double n/t/nocap
#unbuffer python hybrid.py n/vae n/double n/t/nocap

#unbuffer python hybrid.py n/vae n/double n/t/nocap --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/fae n/double n/t/nocap
#
#unbuffer python hybrid.py n/dae n/double n/t/nocap
#
#unbuffer python hybrid.py n/fwae n/double n/t/nocap --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fvae n/double n/t/nocap --model.latent_reg_wt 2


### 47

#unbuffer python hybrid.py n/t/box-s1 --load box-ae_0045-6049543-00_200131-224504/checkpoint_16.pth.tar 
#unbuffer python hybrid.py n/t/box-s1 --load box-wae_0045-6049543-01_200131-224627/checkpoint_16.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load box-vae_0045-6049543-02_200131-224505/checkpoint_16.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load box-vae_0045-6049543-03_200131-224730/checkpoint_16.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load box-fae_0045-6049543-04_200131-224753/checkpoint_16.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load box-fwae_0045-6049543-06_200131-224815/checkpoint_16.pth.tar
#unbuffer python hybrid.py n/t/box-s1 --load box-fvae_0045-6049543-07_200131-224928/checkpoint_16.pth.tar


### 48-49

#unbuffer python hybrid.py n/t/box-s1 --load box-ae_0045-6049543-00_200131-224504/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-wae_0045-6049543-01_200131-224627/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-vae_0045-6049543-02_200131-224505/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-vae_0045-6049543-03_200131-224730/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-fae_0045-6049543-04_200131-224753/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-dae_0045-6049543-05_200131-224815/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-fwae_0045-6049543-06_200131-224815/checkpoint_16.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s1 --load box-fvae_0045-6049543-07_200131-224928/checkpoint_16.pth.tar --model.lr 5e-4

#unbuffer python hybrid.py n/t/box-s1 --load box-dae_0045-6049543-05_200131-224815/checkpoint_16.pth.tar


### 50

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-ae_0047-6049598-01_200201-051024
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-wae_0047-6049598-02_200201-050901
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0047-6049598-03_200201-050901
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0047-6049598-04_200201-051000
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fae_0047-6049598-05_200201-051332
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fwae_0047-6049598-06_200201-051258
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fvae_0047-6049598-07_200201-051256


### 51


#unbuffer python hybrid.py n/ae n/large-double n/celeba
#
#unbuffer python hybrid.py n/wae n/large-double n/celeba
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/large-double n/celeba --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/large-double n/celeba --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/large-double n/celeba --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/large-double n/celeba --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/large-double n/celeba
#
#unbuffer python hybrid.py n/dwae n/large-double n/celeba  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/celeba --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/celeba --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 52-53

#unbuffer python hybrid.py n/ae n/large-double n/atari/spaceinv
#
#unbuffer python hybrid.py n/wae n/large-double n/atari/spaceinv
#
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv
#
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/large-double n/atari/spaceinv --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/large-double n/atari/spaceinv --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/large-double n/atari/spaceinv --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/large-double n/atari/spaceinv
#
#unbuffer python hybrid.py n/dwae n/large-double n/atari/spaceinv  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/spaceinv --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/spaceinv --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 54-55


#unbuffer python hybrid.py n/ae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/wae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/vae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/vae n/large-adain n/celeba --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/large-adain n/celeba --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/large-adain n/celeba --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/large-adain n/celeba --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/large-adain n/celeba --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/large-adain n/celeba
#
#unbuffer python hybrid.py n/dwae n/large-adain n/celeba  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-adain n/celeba --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/large-adain n/celeba --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 56-58

#unbuffer python hybrid.py n/ae n/double n/mpi3d
#
#unbuffer python hybrid.py n/wae n/double n/mpi3d
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/double n/mpi3d --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/double n/mpi3d
#
#unbuffer python hybrid.py n/dwae n/double n/mpi3d  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 59

#unbuffer python hybrid.py --legacy 0 --resume box-s1-dae_0049-6049677-06_200201-144905


### 60

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-ae_0047-6049598-01_200201-051024
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-wae_0047-6049598-02_200201-050901
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0047-6049598-03_200201-050901
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0047-6049598-04_200201-051000
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fae_0047-6049598-05_200201-051332
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fwae_0047-6049598-06_200201-051258
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fvae_0047-6049598-07_200201-051256


### 61

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-ae_0049-6049677-01_200201-144905 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-wae_0049-6049677-02_200201-144902 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0049-6049677-03_200201-144903 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0049-6049677-04_200201-144904 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fae_0049-6049677-05_200201-144901 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fwae_0049-6049677-07_200201-144905 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fvae_0049-6049677-08_200201-144902 --model.lr 5e-4


### 62-63

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0058-6050549-00_200201-230425
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0058-6050549-02_200201-230442
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fwae_0058-6050549-06_200201-230422
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dvae_0058-6050549-08_200201-230424
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dwae_0058-6050549-09_200201-230420
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0058-6050549-10_200201-230427

### 64

#unbuffer python hybrid.py n/ae n/large-double n/atari/spaceinv
#unbuffer python hybrid.py n/wae n/large-double n/atari/spaceinv
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv --model.latent_reg_wt 4
#unbuffer python hybrid.py n/vae n/large-double n/atari/spaceinv --model.latent_reg_wt 16
#unbuffer python hybrid.py n/fwae n/large-double n/atari/spaceinv --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fwae n/large-double n/atari/spaceinv --model.latent_reg_wt 10000 --model.reg_prior 0.01
#unbuffer python hybrid.py n/fvae n/large-double n/atari/spaceinv --model.latent_reg_wt 2
#unbuffer python hybrid.py n/dvae n/large-double n/atari/spaceinv
#unbuffer python hybrid.py n/dwae n/large-double n/atari/spaceinv  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/spaceinv --model.reg_prior 0.1 --model.latent_reg_wt 1000
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/spaceinv --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000

### 65

#unbuffer python hybrid.py --legacy 0 --resume celeba-vae_0051-6050322-02_200201-212740
#unbuffer python hybrid.py --legacy 0 --resume celeba-vae_0051-6050322-04_200201-212724
#unbuffer python hybrid.py --legacy 0 --resume celeba-fwae_0051-6050322-05_200201-212729
#unbuffer python hybrid.py --legacy 0 --resume celeba-fwae_0051-6050322-06_200201-212724
#unbuffer python hybrid.py --legacy 0 --resume celeba-dvae_0051-6050322-08_200201-212722
#unbuffer python hybrid.py --legacy 0 --resume celeba-fdwae_0051-6050322-10_200201-212721
#
#unbuffer python hybrid.py --legacy 0 --resume celeba-ae_0055-6050351-00_200201-215044
#unbuffer python hybrid.py --legacy 0 --resume celeba-wae_0055-6050351-01_200201-215029
#unbuffer python hybrid.py --legacy 0 --resume celeba-fvae_0055-6050351-07_200201-215028
#unbuffer python hybrid.py --legacy 0 --resume celeba-fdwae_0055-6050351-10_200201-215028


### 66

#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.latent_dim 4 --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/gan n/adain-dec n/3dshapes --model.latent_dim 6 --model.generator.splits 1 --model.generator.adain_latent_dim 4


### 67

#unbuffer python hybrid.py n/t/box-s1 --load box-dae_0045-6049543-05_200131-224815/checkpoint_16.pth.tar
#
#unbuffer python hybrid.py n/t/box-s1 --load box-dae_0045-6049543-05_200131-224815/checkpoint_16.pth.tar --model.lr 5e-4


### 68

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-ae_0061-6050842-00_200202-182435 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-wae_0061-6050842-01_200202-182431 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0061-6050842-02_200202-182435 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0061-6050842-03_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fae_0061-6050842-04_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fwae_0061-6050842-05_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fvae_0061-6050842-06_200202-182908 --model.lr 5e-4

### 69

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-ae_0060-6050841-00_200202-182231
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-wae_0060-6050841-01_200202-182239
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0060-6050841-02_200202-182230
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0060-6050841-03_200202-182238
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fae_0060-6050841-04_200202-182235
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fwae_0060-6050841-05_200202-182403
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fvae_0060-6050841-06_200202-182237

### 70-71

#unbuffer python hybrid.py n/ae n/double n/t/nocap
#
#unbuffer python hybrid.py n/wae n/double n/t/nocap
#unbuffer python hybrid.py n/vae n/double n/t/nocap
#
#unbuffer python hybrid.py n/vae n/double n/t/nocap --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/fae n/double n/t/nocap
#
#unbuffer python hybrid.py n/dae n/double n/t/nocap
#
#unbuffer python hybrid.py n/fwae n/double n/t/nocap --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fvae n/double n/t/nocap --model.latent_reg_wt 2


### 72


#unbuffer python hybrid.py n/ae n/double n/mpi3d --dataset.category real
#
#unbuffer python hybrid.py n/wae n/double n/mpi3d --dataset.category real
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --dataset.category real
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 4 --dataset.category real
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 16 --dataset.category real
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 1000 --model.reg_prior 0.1 --dataset.category real
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 10000 --model.reg_prior 0.01 --dataset.category real
#
#unbuffer python hybrid.py n/fvae n/double n/mpi3d --model.latent_reg_wt 2 --dataset.category real
#
#unbuffer python hybrid.py n/dvae n/double n/mpi3d --dataset.category real
#
#unbuffer python hybrid.py n/dwae n/double n/mpi3d  --model.reg_prior 0.1 --model.latent_reg_wt 1000 --dataset.category real
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.latent_reg_wt 1000 --dataset.category real
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000 --dataset.category real


### 73

#unbuffer python hybrid.py --legacy 0 --resume box-s1-dae_0067-6051161-00_200202-194435
#
#unbuffer python hybrid.py --legacy 0 --resume box-s1-dae_0067-6051161-01_200202-194436


### 74

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-ae_0061-6050842-00_200202-182435 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-wae_0061-6050842-01_200202-182431 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0061-6050842-02_200202-182435 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0061-6050842-03_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fae_0061-6050842-04_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fwae_0061-6050842-05_200202-182603 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fvae_0061-6050842-06_200202-182908 --model.lr 5e-4


### 75

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-ae_0060-6050841-00_200202-182231
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-wae_0060-6050841-01_200202-182239
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0060-6050841-02_200202-182230
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-vae_0060-6050841-03_200202-182238
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fae_0060-6050841-04_200202-182235
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fwae_0060-6050841-05_200202-182403
#unbuffer python hybrid.py n/t/box-s3 --load box-s2-fvae_0060-6050841-06_200202-182237


### 76

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-dae_0067-6051161-00_200202-194435

### 77

#unbuffer python hybrid.py n/gan n/adain-dec n/mpi3d --model.latent_dim 4 --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/gan n/adain-dec n/mpi3d --model.latent_dim 6 --model.generator.splits 1 --model.generator.adain_latent_dim 4

#unbuffer python hybrid.py n/gan n/double n/mpi3d --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/mpi3d --model.generator.splits 3 --model.generator.adain_latent_dim 12 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/mpi3d --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.norm_type group --model.up_type bilinear
#
#unbuffer python hybrid.py n/gan n/adain-dec n/mpi3d --model.norm_type group --model.up_type bilinear

### 78

#unbuffer python hybrid.py n/ae n/double n/mpi3d --dataset.category realistic
#
#unbuffer python hybrid.py n/wae n/double n/mpi3d --dataset.category realistic
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --dataset.category realistic
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 4 --dataset.category realistic
#
#unbuffer python hybrid.py n/vae n/double n/mpi3d --model.latent_reg_wt 16 --dataset.category realistic
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 1000 --model.reg_prior 0.1 --dataset.category realistic
#
#unbuffer python hybrid.py n/fwae n/double n/mpi3d --model.latent_reg_wt 10000 --model.reg_prior 0.01 --dataset.category realistic
#
#unbuffer python hybrid.py n/fvae n/double n/mpi3d --model.latent_reg_wt 2 --dataset.category realistic
#
#unbuffer python hybrid.py n/dvae n/double n/mpi3d --dataset.category realistic
#
#unbuffer python hybrid.py n/dwae n/double n/mpi3d  --model.reg_prior 0.1 --model.latent_reg_wt 1000 --dataset.category realistic
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.latent_reg_wt 1000 --dataset.category realistic
#
#unbuffer python hybrid.py n/fdwae n/double n/mpi3d --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000 --dataset.category realistic


### 79

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-dae_0067-6051161-01_200202-194436 --model.lr 5e-4

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-dae_0076-6051205-00_200202-230703

### 80

#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d
#
#unbuffer python hybrid.py n/wae n/adain-dec n/mpi3d
#
#unbuffer python hybrid.py n/vae n/adain-dec n/mpi3d
#
#unbuffer python hybrid.py n/vae n/adain-dec n/mpi3d --model.latent_reg_wt 4
#
#unbuffer python hybrid.py n/vae n/adain-dec n/mpi3d --model.latent_reg_wt 16
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/mpi3d --model.latent_reg_wt 1000 --model.reg_prior 0.1
#
#unbuffer python hybrid.py n/fwae n/adain-dec n/mpi3d --model.latent_reg_wt 10000 --model.reg_prior 0.01
#
#unbuffer python hybrid.py n/fvae n/adain-dec n/mpi3d --model.latent_reg_wt 2
#
#unbuffer python hybrid.py n/dvae n/adain-dec n/mpi3d
#
#unbuffer python hybrid.py n/dwae n/adain-dec n/mpi3d  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/adain-dec n/mpi3d --model.reg_prior 0.1 --model.latent_reg_wt 1000
#
#unbuffer python hybrid.py n/fdwae n/adain-dec n/mpi3d --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 81

#unbuffer python hybrid.py n/t/box-s3 --load box-s2-dae_0079-6051322-00_200203-020049


### 82

#unbuffer python hybrid.py n/t/nocap2cap --load nocap-ae_0071-6051168-00_200202-201832
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-wae_0071-6051168-01_200202-201830
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-02_200202-201832
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-03_200202-201833
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fwae_0071-6051168-06_200202-201831
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fvae_0071-6051168-07_200202-202001


### 83

#unbuffer python hybrid.py --legacy 0 --resume nocap-fae_0071-6051168-04_200202-201833


### 84

#unbuffer python hybrid.py n/t/nocap2cap --load nocap-ae_0071-6051168-00_200202-201832/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-wae_0071-6051168-01_200202-201830/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-02_200202-201832/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-03_200202-201833/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fwae_0071-6051168-06_200202-201831/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fvae_0071-6051168-07_200202-202001/checkpoint_20.pth.tar


### 85

#unbuffer python hybrid.py n/t/nocap2cap --load nocap-ae_0071-6051168-00_200202-201832/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-wae_0071-6051168-01_200202-201830/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-02_200202-201832/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-03_200202-201833/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fwae_0071-6051168-06_200202-201831/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fvae_0071-6051168-07_200202-202001/checkpoint_20.pth.tar --model.lr 5e-4


### 86

#unbuffer python hybrid.py n/t/nocap2cap --load nocap-ae_0071-6051168-00_200202-201832/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-wae_0071-6051168-01_200202-201830/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-02_200202-201832/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-03_200202-201833/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fwae_0071-6051168-06_200202-201831/checkpoint_20.pth.tar
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fvae_0071-6051168-07_200202-202001/checkpoint_20.pth.tar


### 87

#unbuffer python hybrid.py n/t/nocap2cap --load nocap-ae_0071-6051168-00_200202-201832/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-wae_0071-6051168-01_200202-201830/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-02_200202-201832/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-vae_0071-6051168-03_200202-201833/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fwae_0071-6051168-06_200202-201831/checkpoint_20.pth.tar --model.lr 5e-4
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fvae_0071-6051168-07_200202-202001/checkpoint_20.pth.tar --model.lr 5e-4


### 88


#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.generator.splits 2 --model.generator.adain_latent_dim 8
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 4
#
#unbuffer python hybrid.py n/ae n/adain-dec n/3dshapes --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 6


### 89


#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d --model.generator.splits 2 --model.generator.adain_latent_dim 8
#
#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 4
#
#unbuffer python hybrid.py n/ae n/adain-dec n/mpi3d --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 6


### 90


#unbuffer python hybrid.py --legacy 0 --resume celeba-dwae_0055-6050351-09_200201-215029
#
#unbuffer python hybrid.py --legacy 0 --resume celeba-fdwae_0055-6050351-10_200201-215028
#
#unbuffer python hybrid.py --legacy 0 --resume celeba-fdwae_0055-6050351-11_200201-215028


### 91

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-wgan_0077-6051206-02_200202-230936
#
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-wgan_0077-6051206-03_200202-230937


### 92

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0089-6051644-00_200203-125217
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0089-6051644-01_200203-125151
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0089-6051644-02_200203-125158
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0089-6051644-03_200203-125155
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0089-6051644-04_200203-125459


### 93


#unbuffer python hybrid.py --legacy 0 --resume mpi3d-ae_0080-6051330-00_200203-021355
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-wae_0080-6051330-01_200203-021351
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0080-6051330-02_200203-021355
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0080-6051330-03_200203-021351
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0080-6051330-04_200203-021357
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fwae_0080-6051330-05_200203-021352
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fvae_0080-6051330-07_200203-021358
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dvae_0080-6051330-08_200203-021355


### 94

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-wae_0078-6051207-01_200202-231415
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0078-6051207-10_200202-231421
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0078-6051207-11_200202-231547
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0078-6051207-03_200202-231417
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0078-6051207-04_200202-231421
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fwae_0078-6051207-05_200202-231548
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fvae_0078-6051207-07_200202-231549
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dvae_0078-6051207-08_200202-231418
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dwae_0078-6051207-09_200202-231416

### 95

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-wae_0072-6051169-01_200202-201937
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0072-6051169-10_200202-202826
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0072-6051169-02_200202-201938
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0072-6051169-03_200202-201932
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-vae_0072-6051169-04_200202-202104
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fwae_0072-6051169-05_200202-201938
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fvae_0072-6051169-07_200202-201932
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-dwae_0072-6051169-09_200202-202104



### 96

#unbuffer python hybrid.py n/ae n/large-double n/atari/pacman
#unbuffer python hybrid.py n/wae n/large-double n/atari/pacman
#unbuffer python hybrid.py n/vae n/large-double n/atari/pacman
#unbuffer python hybrid.py n/vae n/large-double n/atari/pacman --model.latent_reg_wt 4
#unbuffer python hybrid.py n/vae n/large-double n/atari/pacman --model.latent_reg_wt 16
#unbuffer python hybrid.py n/fwae n/large-double n/atari/pacman --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fwae n/large-double n/atari/pacman --model.latent_reg_wt 10000 --model.reg_prior 0.01
#unbuffer python hybrid.py n/fvae n/large-double n/atari/pacman --model.latent_reg_wt 2
#unbuffer python hybrid.py n/dvae n/large-double n/atari/pacman
#unbuffer python hybrid.py n/dwae n/large-double n/atari/pacman  --model.reg_prior 0.1 --model.latent_reg_wt 1000
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/pacman --model.reg_prior 0.1 --model.latent_reg_wt 1000
#unbuffer python hybrid.py n/fdwae n/large-double n/atari/pacman --model.reg_prior 0.1 --model.prob 0.1 --model.latent_reg_wt 1000


### 97-98

#unbuffer python hybrid.py --legacy 0 --resume spaceinv-ae_0064-6050849-00_200202-183950
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-wae_0064-6050849-01_200202-190020
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-fdwae_0064-6050849-10_200202-190944
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-vae_0064-6050849-02_200202-190121
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-vae_0064-6050849-03_200202-190357
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-fwae_0064-6050849-05_200202-190945
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-fwae_0064-6050849-06_200202-191020
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-fvae_0064-6050849-07_200202-190945
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-dvae_0064-6050849-08_200202-190947
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-dwae_0064-6050849-09_200202-190951


### 99


#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fae_0071-6051168-04_200202-201833/checkpoint_20.pth.tar
#
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-dae_0071-6051168-05_200202-201834/checkpoint_20.pth.tar
#
#
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-fae_0071-6051168-04_200202-201833/checkpoint_20.pth.tar  --model.lr 5e-4
#
#unbuffer python hybrid.py n/t/nocap2cap --load nocap-dae_0071-6051168-05_200202-201834/checkpoint_20.pth.tar  --model.lr 5e-4


### 100-101

#unbuffer python hybrid.py --legacy 0 --resume 3dshapes-ae_0088-6051613-02_200203-121541

### 102

#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0058-6050549-11_200201-230432
#
#unbuffer python hybrid.py --legacy 0 --resume mpi3d-fdwae_0072-6051169-10_200202-202826
#
#unbuffer python hybrid.py --legacy 0 --resume pacman-dwae_0096-6051995-09_200203-152918
#
#unbuffer python hybrid.py --legacy 0 --resume spaceinv-fdwae_0064-6050849-10_200202-190944


### 103

#unbuffer python hybrid.py n/fae n/large-double n/celeba
#
#unbuffer python hybrid.py n/dae n/large-double n/celeba


### 104

#unbuffer python hybrid.py n/ae n/double n/mpi3d --dataset.category realistic

### 105-107


#unbuffer python main_eval_final.py 3dshapes-wae_0028-6046939-01_200131-124832 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-fwae_0051-6050322-06_200201-212724 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-vae_0074-6051172-02_200202-205747 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0080-6051330-04_200203-021357 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-vae_0086-6051608-03_200203-120313 /home/fleeb/results/final
#unbuffer python main_eval_final.py spaceinv-fwae_0064-6050849-05_200202-190945 /home/fleeb/results/final


### 108-109

#unbuffer python main_eval_final.py 3dshapes-vae_0028-6046939-02_200131-124807 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-vae_0028-6046939-03_200131-124821 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-fwae_0028-6046939-05_200131-124815 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-fvae_0029-6046947-06_200131-125026 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-ae_0030-6046949-00_200131-125923 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-dae_0030-6046949-02_200131-125904 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-fae_0031-6046954-00_200131-132552 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-ae_0035-6048636-02_200131-185834 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-wae_0035-6048636-03_200131-185353 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-vae_0035-6048636-04_200131-185348 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-vae_0035-6048636-05_200131-185514 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-dae_0035-6048636-06_200131-185348 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-fae_0035-6048636-07_200131-185701 /home/fleeb/results/final
#unbuffer python main_eval_final.py 3dshapes-fvae_0035-6048636-08_200131-185352 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-ae_0051-6050322-00_200201-212755 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-wae_0051-6050322-01_200201-212724 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-vae_0051-6050322-02_200201-212740 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-vae_0051-6050322-03_200201-212735 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-vae_0051-6050322-04_200201-212724 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-fwae_0051-6050322-05_200201-212729 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-fvae_0051-6050322-07_200201-212728 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-ae_0055-6050351-00_200201-215044 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-wae_0055-6050351-01_200201-215029 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-fwae_0055-6050351-05_200201-215036 /home/fleeb/results/final
#unbuffer python main_eval_final.py celeba-fwae_0055-6050351-06_200201-215029 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-ae_0058-6050549-00_200201-230425 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-wae_0058-6050549-01_200201-230423 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fdwae_0058-6050549-10_200201-230427 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0058-6050549-02_200201-230442 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0058-6050549-03_200201-230427 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0058-6050549-04_200201-230424 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0058-6050549-05_200201-230421 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0058-6050549-06_200201-230422 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-dvae_0058-6050549-08_200201-230424 /home/fleeb/results/final
#unbuffer python main_eval_final.py spaceinv-wae_0064-6050849-01_200202-190020 /home/fleeb/results/final
#unbuffer python main_eval_final.py spaceinv-vae_0064-6050849-03_200202-190357 /home/fleeb/results/final
#unbuffer python main_eval_final.py spaceinv-fvae_0064-6050849-07_200202-190945 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-ae_0072-6051169-00_200202-201932 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-wae_0072-6051169-01_200202-201937 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0072-6051169-02_200202-201938 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0072-6051169-03_200202-201932 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0072-6051169-04_200202-202104 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0072-6051169-05_200202-201938 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0072-6051169-06_200202-202104 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fvae_0072-6051169-07_200202-201932 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-ae_0074-6051172-00_200202-205115 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-wae_0074-6051172-01_200202-205635 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-vae_0074-6051172-02_200202-205747 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-vae_0074-6051172-03_200202-211833 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-fae_0074-6051172-04_200202-211819 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-fwae_0074-6051172-05_200202-211800 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-fvae_0074-6051172-06_200202-211800 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-wae_0078-6051207-01_200202-231415 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0078-6051207-02_200202-231418 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0078-6051207-03_200202-231417 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0078-6051207-04_200202-231421 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0078-6051207-05_200202-231548 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0078-6051207-06_200202-231421 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fvae_0078-6051207-07_200202-231549 /home/fleeb/results/final
#unbuffer python main_eval_final.py box-s3-dae_0079-6051322-01_200203-020055 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-ae_0080-6051330-00_200203-021355 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-wae_0080-6051330-01_200203-021351 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0080-6051330-02_200203-021355 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-vae_0080-6051330-03_200203-021351 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fwae_0080-6051330-05_200203-021352 /home/fleeb/results/final
#unbuffer python main_eval_final.py mpi3d-fvae_0080-6051330-07_200203-021358 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-ae_0086-6051608-00_200203-120340 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-wae_0086-6051608-01_200203-120307 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-vae_0086-6051608-02_200203-120325 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-fwae_0086-6051608-04_200203-120305 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-fvae_0086-6051608-05_200203-120438 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-fae_0099-6052165-00_200203-161155 /home/fleeb/results/final
#unbuffer python main_eval_final.py nocap2cap-dae_0099-6052165-01_200203-161022 /home/fleeb/results/final


###





#######


#unbuffer python hybrid.py n/ae n/large-adain n/celeba --model.generator.splits 3 --model.generator.adain_latent_dim 12
#
#unbuffer python hybrid.py n/ae n/large-adain n/celeba --model.generator.splits 1 --model.generator.adain_latent_dim 4
#
#unbuffer python hybrid.py n/ae n/large-adain n/celeba --model.generator.splits 2 --model.generator.adain_latent_dim 8
#
#unbuffer python hybrid.py n/ae n/large-adain n/celeba --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 4
#
#unbuffer python hybrid.py n/ae n/large-adain n/celeba --model.generator.splits 1 --model.generator.adain_latent_dim 4 --model.latent_dim 6




###

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-dae_0049-6049677-06_200201-144905

#unbuffer python hybrid.py n/t/box-s2 --load box-s1-ae_0049-6049677-01_200201-144905 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-wae_0049-6049677-02_200201-144902 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0049-6049677-03_200201-144903 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-vae_0049-6049677-04_200201-144904 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fae_0049-6049677-05_200201-144901 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fwae_0049-6049677-07_200201-144905 --model.lr 5e-4
#unbuffer python hybrid.py n/t/box-s2 --load box-s1-fvae_0049-6049677-08_200201-144902 --model.lr 5e-4




###

#unbuffer python hybrid.py --legacy 0 --resume

#unbuffer python hybrid.py n/ae n/double n/t/box
#unbuffer python hybrid.py n/wae n/double n/t/box
#unbuffer python hybrid.py n/vae n/double n/t/box
#unbuffer python hybrid.py n/vae n/double n/t/box --model.latent_reg_wt 4
#unbuffer python hybrid.py n/fae n/double n/t/box
#unbuffer python hybrid.py n/fwae n/double n/t/box --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fvae n/double n/t/box --model.latent_reg_wt 2
#unbuffer python hybrid.py n/dae n/double n/t/box



#unbuffer python hybrid.py n/ae n/large-adain n/3dshapes


#unbuffer python hybrid.py n/wae n/large-adain n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#
#unbuffer python hybrid.py n/wae n/large-adain n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-adain n/3dshapes
#
#unbuffer python hybrid.py n/vae n/large-adain n/3dshapes --model.latent_reg_wt 4



### std autoenc


#unbuffer python hybrid.py n/wae n/double n/3dshapes --model.latent_reg_wt 0 --info.model_type ae
#unbuffer python hybrid.py n/wae n/double n/3dshapes

#unbuffer python hybrid.py n/vae n/double n/3dshapes
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.latent_reg_wt 4
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.latent_reg_wt 16

#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1
#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 10000 --model.reg_prior 0.01

#unbuffer python hybrid.py n/fvae n/double n/3dshapes --model.latent_reg_wt 2



### fast + dae


#unbuffer python hybrid.py n/ae n/double n/3dshapes
#
#unbuffer python hybrid.py n/wae n/double n/3dshapes
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes
#
#unbuffer python hybrid.py n/vae n/double n/3dshapes --model.latent_reg_wt 4

#unbuffer python hybrid.py n/fae n/double n/3dshapes

#unbuffer python hybrid.py n/dae n/double n/3dshapes

#unbuffer python hybrid.py n/fwae n/double n/3dshapes --model.latent_reg_wt 1000 --model.reg_prior 0.1

#unbuffer python hybrid.py n/fvae n/double n/3dshapes --model.latent_reg_wt 2










