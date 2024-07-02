from models import cvae,cgan,pkgan,pkvae,pkvae_gank


class CoreModel:
    def __init__(self, device, args, n_class=5):
        self.device = device
        self.args = args
        self.model_name = args.model
        self.n_class = n_class

    def get(self):
        if self.model_name == 'cvae':
            return self.get_fcn()
        elif self.model_name == 'cgan':
            return DAFormer('./mmseg/configs/daformer/daformer_mit5.py')
        elif self.model_name == 'cvaegan':
            return SegCLIP(train_clip=(not self.args.freeze_clip))
        else:
            return None

    def get_fcn(self):
        vgg_model = VGGNet(requires_grad=True, remove_fc=True).to(self.device)
        fcn_model = FCNs(pretrained_net=vgg_model, n_class=self.n_class, p=self.args.drop).to(self.device)
        return fcn_model


if __name__ == '__main__':
    print(1)