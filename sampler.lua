--[[
   VAE: Gaussian Sampler / Reparameterization trick

   @author: Sagar M. Waghmare
--]]

-- Sampler
dofile('GaussianNoise.lua')

-- input is log(Var) i.e log(Sigma^2) = 2 * log(Sigma)
local getSigma = nn.Sequential()
getSigma:add(nn.MulConstant(0.5)) -- 0.5 * 2 * log(sigma) -> log(Sigma)
getSigma:add(nn.Exp()) -- exp (log(Sigma)) -> Sigma

-- Sigma and Noise
local getSigmaNoise = nn.ConcatTable()
getSigmaNoise:add(getSigma) -- Sigma
getSigmaNoise:add(nn.GaussianNoise()) -- Noise

-- Sigma*Noise
local scaleSigma = nn.Sequential()
scaleSigma:add(getSigmaNoise):add(nn.CMulTable()) -- Sigma*Soise

-- Mean and Sigma*Noise
local getMeanScaledSigma = nn.ParallelTable()
getMeanScaledSigma:add(nn.Identity()) -- Mean
getMeanScaledSigma:add(scaleSigma) -- Sigma*Noise

-- Mean + Sigma*Noise
local sampler = nn.Sequential()
sampler:add(getMeanScaledSigma):add(nn.CAddTable())

return sampler
