fn = "diags/reducedfiles/DiffLumi_ele1_pos2.txt"
with open(fn) as f:
    header = f.readline()
print(header[:400])
print("...", header[-300:])
E, d = extract_Ecom_and_dLdE(fn)
print(len(E), len(d), np.loadtxt(fn).shape)