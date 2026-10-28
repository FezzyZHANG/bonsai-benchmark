src=open("/workspace/tools/simfinal.py").read()
g={}
exec(src,g)
rn=g["rn"]; t=g["t"]; S2560=g["S2560"]; H=g["H"]
print("rn first3",rn[:3],"rn sum",rn.sum(),"oracle rn first [18.5188,4.0349,-3.1824] sum -183.335")
sg=S2560*rn
print("signed first3",sg[:3],"sum",sg.sum(),"oracle -18.5188,-4.0349,3.1824 sum 27.0427")
print("t row0 first3",t[:3],"last3",t[509:512],"sum",t.sum(),"oracle [0.0019,-2.1346,-1.1733] [-0.5485,2.2623,5.9474] sum -337.271")
