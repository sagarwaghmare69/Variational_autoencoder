--[[
   Implementation of Variational Autoencoder.

   Ref: https://arxiv.org/abs/1312.6114
   Ref: https://arxiv.org/abs/1606.05908 (Tutorial)
  
   @author: Sagar M. Waghmare
--]]

require 'math'

torch.setdefaulttensortype("torch.FloatTensor")

cmd = torch.CmdLine()
cmd:text()
cmd:text()
local titleMsg = 'Variational Autoencoder for MNIST.'
cmd:text(titleMsg)
cmd:text()
cmd:text()

-- Data
-- Using MNIST datset from dp

-- Model parameters
cmd:option('--noiseSigma', 0,
           'Stdev for noise for denoising autoencoder (Mean is zero).')
cmd:option('--hiddens', '{1000, 500, 250, 250, 250}', 'Hiddens units')
cmd:option('--latentDim', 250, 'Hiddens units')
cmd:option('--useBatchNorm', false, 'Use batch normalization')
cmd:option('--activation', 'ReLU', 'Non linearity')

-- Criterion and learning
cmd:option('--batchSize', 32, 'Batch Size.')
cmd:option('--epochs', 2000, 'Number of epochs.')
cmd:option('--learningRate', 0.002, 'Learning rate')
cmd:option('--learningRateDecay', 1e-07, 'Learning rate decay')
cmd:option('--momentum', 0, 'Learning Momemtum')
cmd:option('--adam', false, 'Use adaptive moment estimation optimizer.')

-- Use Cuda
cmd:option('--useCuda', false, 'Use GPU')
cmd:option('--deviceId', 1, 'GPU device Id')

-- Print debug messages
cmd:option('--verbose', false, 'Print apppropriate debug messages.')

-- Command line arguments
opt = cmd:parse(arg)
print(opt)

-- Fixing seed to faciliate reproduction of results.
torch.manualSeed(-1)

verbose = opt.verbose

--MNIST datasource
ds, trData, tvData, tsData = dofile('datasource.lua')
if verbose then
   print(trData)
   print(tvData)
   print(tsData)
end

encoder, sampler = dofile('model.lua')

