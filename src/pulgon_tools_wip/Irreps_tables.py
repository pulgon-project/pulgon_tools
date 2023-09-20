import numpy as np
import matplotlib.pyplot as plt
from  pdb import set_trace
import pretty_errors


def _cal_irrep_trace(irreps, symprec):
    character_tabel = []
    for irrep in irreps:
        tmp_character = []
        for tmp in irrep:

            if tmp.size == 1:
                if abs(tmp.imag) < symprec:
                    tmp = tmp.real
                if abs(tmp.real) < symprec:
                    tmp = complex(0, tmp.imag)
                    if tmp.imag == 0:
                        tmp = 0
                tmp_character.append(tmp)
            else:
                tmp = np.trace(tmp)
                if abs(tmp.imag) < symprec:
                    tmp = tmp.real
                if abs(tmp.real) < symprec:
                    tmp = complex(0, tmp.imag)
                    if abs(tmp.imag) < symprec:
                        tmp = 0
                tmp_character.append(tmp)
        character_tabel.append(tmp_character)
    return character_tabel


def line_group_1(q: int, r: int, a: float, f: float, n: int, m1: int,k1: float, m2: int, k2: float, symprec: float = 1e-4):
    """ TQ(f)Cn """
    # label for line group family
    row_labels = [r"$(C_{Q}|f)$", r"$C_{n}$"]
    column_labels = [r"$_{k}A_{m}$", r"$_{\widetilde{k}}A_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if k1<=-np.pi/a or k1>np.pi/a:
        judge = False
        message.append("k1 not belong to (-pi/a,pi/a]")
    if m1<=-q/2 or m1>q/2:
        judge = False
        message.append("m1 not belong to (-q/2,q/2]")
    if k2<=-np.pi/f or k2>np.pi/f:
        judge = False
        message.append("k2 not belong to (-pi/f,pi/f]")
    if m2<=-n/2 or m2>n/2:
        judge = False
        message.append("m1 not belong to (-n/2,n/2]")

    if judge:
        irrep1 = [np.exp(1j*(k1*f+m1*2*np.pi*r/q)), np.exp(1j*m1*2*np.pi/n)]
        irrep2 = [np.exp(1j*k2*f), np.exp(1j*m2*2*np.pi/n)]
        irreps = np.round([irrep1, irrep2], 4)
        character_tabel = _cal_irrep_trace(irreps, symprec)
        return character_tabel, row_labels, column_labels
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_2(a: float, n: float, k1: float, m1: int, k2: float, m2:int, sigmah: int, symprec: float =1e-4) -> [list, list, list]:
    """ T(a)S2n """
    row_labels = [r"$(I|a)$", r"$\sigma_{h}C_{2n}$"]
    column_labels = [r"$_{k}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1)>=symprec and abs(k1 - np.pi/a)>symprec:
        judge = False
        message.append("k1 not equal to 0 or pi/a")
    if m1<=-n/2 or m1>n/2:
        judge = False
        message.append("m1 not belong to (-n/2,n/2]")
    if k2<=0 or k2>=np.pi:
        judge = False
        message.append("k2 not belong to (0,pi)")
    if m2<=-n/2 or m2>n/2:
        judge = False
        message.append("m1 not belong to (-n/2,n/2]")

    if judge:
        irrep1 = np.round([np.exp(1j*k1*a), sigmah * np.exp(1j*m1*np.pi/n)], 4)
        irrep2 = np.round([[[np.exp(1j*k2*a), 0], [0, np.exp(-1j*k2*a)]], [[0, np.exp(1j*m2*np.pi/n)], [1, 0]]], 4)
        irreps = [irrep1, irrep2]
        character_tabel = _cal_irrep_trace(irreps, symprec)
        return character_tabel, row_labels, column_labels
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_3(a: float, n: float, k1: float, m1: int, k2: float, m2:int, sigmah: int, symprec: float =1e-4) -> [list, list, list]:
    """ T(a)Cnh """
    row_labels = [r"$(I|a)$", r"$C_{n}$", r"$\sigma_{h}$"]
    column_labels = [r"$_{k}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1)>=symprec and abs(k1 - np.pi/a)>symprec:
        judge = False
        message.append("k1 not equal to 0 or pi/a")
    if m1<=-n/2 or m1>n/2:
        judge = False
        message.append("m1 not belong to (-n/2,n/2]")
    if k2<=0 or k2>=np.pi:
        judge = False
        message.append("k2 not belong to (0,pi)")
    if m2<=-n/2 or m2>n/2:
        judge = False
        message.append("m1 not belong to (-n/2,n/2]")

    if judge:
        irrep1 = np.round([np.exp(1j*k1*a), np.exp(1j*m1*2*np.pi/n), sigmah], 4)
        irrep2 = np.round([[[np.exp(1j*k2*a), 0], [0, np.exp(-1j*k2*a)]], [[np.exp(1j*m2*2*np.pi/n), 0], [0, np.exp(1j*m2*2*np.pi/n)]], [[0, 1], [1, 0]]], 4)
        irreps = [irrep1, irrep2]
        character_tabel = _cal_irrep_trace(irreps, symprec)
        return character_tabel, row_labels, column_labels
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)


def line_group_4(a: float, n: float, k1: float, m1: int, k2: float, m2:int, k3: float, m3:int, k4: float, m4:int, sigmah: int, symprec: float =1e-4) -> [list, list, list]:
    """ T(a)Cnh """
    row_labels = [r"$(C_{2n}|1/2)$", r"$C_{n}$", r"$\sigma_{h}$"]
    column_labels = [r"$_{0}A_{m}^{\Pi_{h}}$", r"$_{k}E_{m}$", r"$_{\widetilde{k}_{\widetilde{M}}(\widetilde{m})}A_{\widetilde{m}}^{\Pi_{h}}$", r"$_{\widetilde{k}}E_{\widetilde{m}}$"]

    # whether the input satisfy the requirements
    judge = True
    message = []
    if abs(k1)>=symprec:
        judge = False
        message.append("k1 not equal to 0")
    if m1<=-n or m1>n:
        judge = False
        message.append("m1 not belong to (-n,n]")
    if k2<=0 or k2>np.pi/a:
        judge = False
        message.append("k2 not belong to (0,pi/a]")
    if m2<=-n or m2>n:
        judge = False
        message.append("m1 not belong to (-n,n]")
    if judge:
        irrep1 = np.round([np.exp(1j*m1*np.pi/n), np.exp(1j*m1*2*np.pi/n), sigmah], 4)
        irrep2 = np.round([[[np.exp(1j*(m2*np.pi/n+k2*a/2)), 0], [0, np.exp(1j*(m2*np.pi/n-k2*a/2))]], [[np.exp(1j*m2*2*np.pi/n), 0], [0, np.exp(1j*m2*2*np.pi/n)]], [[0, 1], [1, 0]]], 4)
        irrep3 = np.round([np.exp(1j*k3*a/2), np.exp(1j*m3*2*np.pi/n), sigmah], 4)
        irrep4 = np.round([[[np.exp(1j*k4*a/2), 0], [0, np.exp(1j*(m4*2*np.pi/n-k4*a/2))]], [[np.exp(1j*m4*2*np.pi/n), 0], [0, np.exp(1j*m4*2*np.pi/n)]], [[0, 1], [1, 0]]], 4)
        irreps = [irrep1, irrep2, irrep3, irrep4]
        character_tabel = _cal_irrep_trace(irreps, symprec)
        return character_tabel, row_labels, column_labels
    else:
        print("error of input:")
        for tmp in message:
            print(tmp)
        return None, None, None


def plot_character_table(character, row, column):
    plt.figure(dpi=200)
    # fig,ax = plt.subplots(1,2)
    # set_trace()
    plt.axis('off')
    plt.table(cellText=character,
              rowLabels=column,
              colLabels=row,
              colWidths=[0.2 for x in row],
              # colColours=colColors,
              # rowColours=rowColours,
              cellLoc='center',
              rowLoc='center',
              loc="center")
    # ax[1].axis('off')
    # ax[1].text("123456")
    # plt.savefig("fig2.png", dpi=400)
    plt.show()


def main():
    # line 1
    q, r, f, n = 3, 1, 3, 5
    Q = q/r
    a = Q*f
    m1, k1, m2, k2 = 1, 0, 1, 0

    # # line 2
    # a = 3
    # n = 6
    # k1 = np.pi/a
    # m1 = 2
    # k2 = np.pi/3
    # m2 = 0
    # sigmah = -1
    #
    # # line 3
    # a = 3
    # n = 6
    # k1 = np.pi/a
    # m1 = 2
    # k2 = np.pi/3
    # m2 = 0
    # sigmah = -1
    #
    # # line 4
    # a = 3
    # n = 6
    # k1 = 0
    # m1 = 2
    # k2 = np.pi/a
    # m2 = 0
    # k3 = np.pi/a
    # m3 = 1
    # k4 = np.pi/a
    # m4 = -1
    # sigmah = -1

    character_tabel, row_labels, column_labels = line_group_1(q, r, a, f, n, m1, k1, m2, k2)
    # character_tabel, row_labels, column_labels = line_group_4(a, n, k1, m1, k2, m2, k3, m3, k4, m4, sigmah)
    if character_tabel!=None:
        plot_character_table(character_tabel, row_labels, column_labels)


if __name__ == '__main__':
    main()
