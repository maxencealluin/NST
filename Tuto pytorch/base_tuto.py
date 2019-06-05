import torch
#
# x = torch.empty(5, 3)
# print(x)
#
# x = torch.rand(5, 3)
# print(x)
#
# x = torch.zeros(5, 3, dtype = torch.long)
# print(x)
#
# x = torch.tensor([5,3])
# print(x)
#
# x = torch.ones(5 , 3, dtype=torch.double)
# print(x)
#
# x = torch.randn_like(x, dtype=torch.float)
# print(x)

# x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# e = torch.empty(5, 3)
# print (e.size());


# torch.add(x, y, out = e)
# print(e)
# print (e[0,:])

# x = torch.randn(4,4)
# print(x)
# print(x.size())
# x = x.view(x.size()[0] * x.size()[1])
# print(x)
# x = x.view(2,8)
# print(x)
# x = x.view(-1,4)
# print(x)
#
# x = torch.randn(1)
# print(x)
# print(x.item())

# #numpy compatibility
# a = torch.ones(5,2)
# print(a)
#
# b = a.numpy()
# print(b)
# a.add_(1)
#
# # = tensor du coup array liee
#
# print(a)
# print(b)

#AUTOGRAD

# x = torch.ones(2,2, requires_grad=True)
# # # print(x)
# # #
# y = x + 2
# # # print(y)
# # print(y.grad_fn)
# z = y * y * 3
# out = z.mean()
# # print(z, "\n\n", out)
# #
# a = torch.randn(2, 2)
# a = (a * 3) / (a - 1)
# # print(a.requires_grad)
# a.requires_grad = True
# # print(a.requires_grad)
# b = (a * a).sum()
# # print(b.grad_fn)
# print(out)
# print("grad:", x.grad)
# out.backward()
# print("grad:", x.grad)

# x = torch.randn(3, requires_grad=True)
#
# y = x * 2
# while y.data.norm() < 1000:
# 	y = y * 2
#
# print(y)
# print(y.norm())
#
# v = torch.tensor([0.1, 1.0, 0.0001], dtype = torch.float)
# y.backward(v)
# print(x.grad)
#
# print(x.requires_grad)
# print((x ** 2).requires_grad)
#
# with torch.no_grad():
# 	print((x ** 2).requires_grad)

lp_filter = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]], dtype= torch.float, device ='cuda')
lp_filter = lp_filter.view(1, 1, 3, 3).repeat(1, 3, 1, 1)
print(lp_filter)
print(lp_filter.shape)