local GaussianNoise, Parent = torch.class('nn.GaussianNoise', 'nn.Module')

function GaussianNoise:__init(mean, std)
   Parent.__init(self)
   -- std corresponds to 50% for MNIST training data std.
   self.mean = mean or 0
   self.std = std or 1
end

function GaussianNoise:updateOutput(input)
   self.output:resizeAs(input)
   self.output:normal(self.mean, self.std)
   return self.output
end

-- Non continuous Operation -> No Gradients
function GaussianNoise:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):zero()
   return self.gradInput
end

function GaussianNoise:__tostring__()
  return string.format('%s mean: %f, std: %f', 
                        torch.type(self), self.mean, self.std)
end
