import numpy as np
from scipy import stats
import swyft
import torch
import scipy.ndimage
from PIL import Image, ImageFont, ImageDraw
import scipy.ndimage as ndimage



class SimulatorLinePattern(swyft.Simulator):
    def __init__(self, Npix = 256, bounds = None, sigma = 0., randn_realizations = None):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        g = np.linspace(-1, 1, Npix)
        self.grid = np.meshgrid(g, g)
        self.z_sampler = swyft.RectBoundSampler([
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(-1, 2),
            stats.uniform(0.1, 0.3)
        ], bounds = bounds)
        self.sigma = sigma
        self.Npix = Npix
        self.randn_realizations = randn_realizations
        
    def templates(self, z):
        w = z[4]
        sigmoid = lambda x: 1/(1+np.exp(-x*100))
        t1 = np.exp(-np.sin((self.grid[0]-z[0]+self.grid[1]-z[1])/0.02)**2/0.2)
        m1 = sigmoid(self.grid[0]-(z[0]-w))*sigmoid(-self.grid[0]+z[0]+w)*sigmoid(self.grid[1]-z[1]+w)*sigmoid(-self.grid[1]+z[1]+w)
        t2 = np.exp(-np.sin((self.grid[1]-z[3])/0.016)**2/0.4)
        m2 = sigmoid(self.grid[0]-z[2]+w)*sigmoid(-self.grid[0]+z[2]+w)*sigmoid(self.grid[1]-z[3]+w)*sigmoid(-self.grid[1]+z[3]+w)
        return t1*m1 + t2*m2
    
    def sample_noise(self):
        if self.randn_realizations is None:
            return np.random.randn(self.Npix, self.Npix)*self.sigma
        else:
            i, j, k = np.random.randint((len(self.randn_realizations), 256, 256))
            r = self.randn_realizations[i]*self.sigma
            r = np.roll(np.roll(r, j, axis = 0), k, axis = 1)
            return r
    
    def build(self, graph):
        z = graph.node('z', self.z_sampler)
        mu = graph.node('mu', self.templates, z)
        x = graph.node('x', lambda mu: mu + self.sample_noise(), mu)
        
        
class SimulatorBlob(swyft.Simulator):
    def __init__(self, bounds = None, Npix = 64, sigma = 0.1):
        super().__init__()
        self.transform_samples = swyft.to_numpy32
        self.bounds = bounds
        self.weights, self.dist, self.Cov = self.setup_cov(Npix = Npix)
        self.Npix = Npix
        self.sigma = sigma

    def setup_cov(self, Npix = 64):
        N = Npix**2
        l = torch.linspace(-1, 1, Npix)
        L1, L2 = torch.meshgrid(l, l)
        L = torch.stack([L1, L2], dim = -1)
        T = L.unsqueeze(0).unsqueeze(0) - L.unsqueeze(2).unsqueeze(2)
        T = (T**2).sum(-1)**0.5
        T = T.view(N, N)
        Cov = torch.exp(-T/.5)*0.5 + torch.exp(-T/0.5)*.25 + torch.exp(-T/2)*.125
        Cov *= 2
        dist = torch.distributions.MultivariateNormal(torch.zeros(N), Cov)
        R = (L1**2+L2**2)**0.5
        weights = torch.exp(-0.5*(R-0.5)**2/0.1**2)*0 + 1
        return weights, dist, Cov
        
    def sample_GP(self):
        if self.bounds is None:
            return self.dist.sample(torch.Size([1]))[0].numpy().reshape(self.Npix, self.Npix)
        else:
            i = np.random.randint(len(self.bounds))
            return self.bounds[i]
            
    def build(self, graph):
        z = graph.node("z", lambda: self.sample_GP())
        zn = graph.node("zn", lambda z: z + np.random.randn(self.Npix, self.Npix)*self.sigma, z)
        mu = graph.node("mu", lambda z: self.weights*np.exp(z*0.5)*0.5 , z)
        

class SimulatorLetter(swyft.Simulator):
    def __init__(self, bounds = None):
        super().__init__()
        self.letter_imgs = [self.get_letter(l) for l in "AbCdEf"]
        self.transform_samples = swyft.to_numpy32
        self.bounds = bounds

    # set font and font size, from ChatGPT
    @staticmethod
    def get_letter(l):
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf', 50)
        img = Image.new('RGB', (48, 48), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((9, -5), l, fill='black', font=font)
        arr = np.array(img)[:,:,0]
        arr = 1-arr/arr.max()
        return arr
    
    @staticmethod
    def add_letter(img, rot, pos, output_shape):
        img = scipy.ndimage.rotate(img, rot, mode = 'constant', reshape = False)
        mapping = lambda x: (x[0]-pos[0]*output_shape[0], x[1]-pos[1]*output_shape[1])
        img = scipy.ndimage.affine_transform(img, np.eye(2), offset = -pos*256, output_shape = (256, 256))
        return img

    def gen_letter_sequence(self, rot, pos):
        img = np.zeros((256, 256))
        for i in range(6):
            img += self.add_letter(self.letter_imgs[i], rot[i], pos[i]*0.85, (256, 256))
        return img
    
    def get_pos(self):
        if self.bounds is None:
            return np.random.rand(6, 2)
        else:
            i = np.random.randint(len(self.bounds[0]))
            return self.bounds[:,i,:]
    
    def build(self, graph):
        rot = graph.node("rot", lambda: np.random.rand(6)*0)
        pos = graph.node('pos', self.get_pos)
        mu = graph.node('mu', lambda rot, pos: self.gen_letter_sequence(rot, pos), rot, pos)
        
        
class SimulatorSnake(swyft.Simulator):
    def __init__(self, bounds = None, bounds_n_steps = None, n_max = 50):
        super().__init__()
        self.bounds = bounds
        self.n_max = n_max
        self.bounds_n_steps = bounds_n_steps
        self.transform_samples = swyft.to_numpy32
        
    def get_image(self, turns, n_steps):
        turns = turns[:n_steps[0]]
        directions_angle = np.cumsum(turns)*np.pi/2
        directions_cart = np.array([np.cos(directions_angle), np.sin(directions_angle)]).T
        path = np.array([np.cumsum(directions_cart[:,0]), np.cumsum(directions_cart[:,1])]).T
        img = np.zeros((256, 256))
        idxs = np.floor(path*4).astype(int)+128
        for i, j in idxs:
            if i < 0 or j < 0 or i > 255 or j > 255:
                pass
            else:
                img[i, j] += 1
        img = ndimage.gaussian_filter(img, sigma=1, order=0)
        return img
                
    def get_n_steps(self):
        b = self.bounds_n_steps
        if b is not None:
            return np.random.randint(b[0], b[1]+1, size = (1,))
        else:
            return np.random.randint(10, self.n_max, size = (1,))
    
#    def get_turns(self):
#        turns = np.random.choice([-1, 0, 1], self.n_max, p = [0.1, 0.8, 0.1])
#        return turns
    
    def get_turns(self):
        probs = np.array([[0.1, 0.8, 0.1]]).repeat(self.n_max, axis=0)
        bounds = self.bounds
        if bounds is not None:
            probs = probs*bounds
        indxs = (probs.cumsum(1)/probs.sum(1)[:,None] > np.random.rand(probs.shape[0])[:,None]).argmax(1)
        choices = np.array([-1., 0., 1.])
        return choices[indxs]

    def build(self, graph):
        n_steps = graph.node("n_steps", self.get_n_steps)
        turns = graph.node('turns', self.get_turns)
        img = graph.node('mu', self.get_image, turns, n_steps)
        
        
class SimulatorUnraveling(swyft.Simulator):
    def __init__(self, sigma = 2.0):
        super().__init__()
        self.sigma = sigma
        self.transform_samples = swyft.to_numpy32
        
    def gen_image(self, pos):        
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf', 22)
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        x, y = pos
        draw.text((x+10, y+10), "Unvraveling", fill='black', font=font)
        draw.text((x+35, y+35), "the", fill='black', font=font)
        draw.text((x+60, y+60), "Un-Unravelable", fill='black', font=font)
        arr = np.array(img)[:,:,0]
        arr = 1-arr/arr.max()
        return arr
    
    def gen_pos(self):
        x, y = np.random.rand(2)*256-128
        y += 75
        return np.array([x, y])
    
    def build(self, graph):
        pos = graph.node("pos", self.gen_pos)
        mu = graph.node('mu', self.gen_image, pos)
        x = graph.node('x', lambda mu: mu + np.random.randn(*mu.shape)*self.sigma, mu)
