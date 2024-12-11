#include <string>
#include <vector>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include <cstdlib>
#include <sys/stat.h>
#include <fcntl.h>
using namespace std;

#define __1U1B__

int main(int argc, char **argv)
{
    vector<string> mlist = {
        "drugbank1.pdbqt",
        "drugbank2.pdbqt",
        "drugbank3.pdbqt",
        "drugbank4.pdbqt",
        "drugbank5.pdbqt",
        "drugbank6.pdbqt",
        "drugbank7.pdbqt",
        "drugbank8.pdbqt",
        "drugbank9.pdbqt",
        "drugbank10.pdbqt",
        "drugbank11.pdbqt",
        "drugbank12.pdbqt",
        "drugbank13.pdbqt",
        "drugbank14.pdbqt",
        "drugbank15.pdbqt",
        "drugbank16.pdbqt",
        "drugbank17.pdbqt",
        "drugbank18.pdbqt",
        "drugbank19.pdbqt",
        "drugbank20.pdbqt",
        "drugbank21.pdbqt",
        "drugbank22.pdbqt",
        "drugbank23.pdbqt",
        "drugbank24.pdbqt",
        "drugbank25.pdbqt",
        "drugbank26.pdbqt",
        "drugbank27.pdbqt",
        "drugbank28.pdbqt",
        "drugbank29.pdbqt",
        "drugbank30.pdbqt",
    };

#ifdef __2ER7__
    string dir = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/test/";
#endif

#ifdef __2BM2__
    string dir = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/test/";
#endif

#ifdef __1U1B__
    string dir = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/test/";
#endif

    const char *myargv[] = {
        "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/Vina-GPU-2-1-CUDA",
        "--ligand", "",
        "--thread", "8192",
#ifdef __2ER7__
        "--center_x", "2.865",
        "--center_y", "30.924",
        "--center_z", "17.484",
        "--size_x", "50",
        "--size_y", "26.25",
        "--size_z", "22.5",
        "--receptor", "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/2er7_protein.pdbqt",
#endif

#ifdef __2BM2__
        "--center_x", "40.415",
        "--center_y", "110.986",
        "--center_z", "82.673",
        "--size_x", "29",
        "--size_y", "29",
        "--size_z", "29",
        "--receptor", "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/2bm2_protein.pdbqt",
#endif

#ifdef __1U1B__
        "--center_x", "-30.661",
        "--center_y", "-29.540",
        "--center_z", "17.010",
        "--size_x", "24.0",
        "--size_y", "26.25",
        "--size_z", "22.5",
        "--receptor", "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/1u1b_protein.pdbqt",
#endif
        NULL};

    pid_t pid;
    int wstatus;
    string s;

#ifdef __2ER7__
    string log = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/log_2er7.txt";
#endif

#ifdef __2BM2__
    string log = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/log_2bm2.txt";
#endif

#ifdef __1U1B__
    string log = "/mnt/disk4/DingXiaoyu/AutoDock-Vina-GPU-2-1-CUDA-V5-LCF/auto_test/log_1u1b.txt";
#endif

    int fd = open(log.data(), O_WRONLY | O_CREAT | O_APPEND, 0644);

    int i = 0;
    while (i < mlist.size())
    {
        s = dir + mlist[i];
        myargv[2] = s.data();

        if ((pid = fork()) < 0)
        {
            cout << "fork error" << endl;
            return 0;
        }

        if (pid > 0)
        {
            waitpid(pid, &wstatus, 0);
            cout << "finish " << myargv[2] << endl;
        }
        else
        {
            if (fd < 0)
            {
                cout << "open error" << endl;
                exit(1);
            }
            if (dup2(fd, STDOUT_FILENO) < 0)
            {
                cout << "dup2 error" << endl;
                exit(1);
            }
            close(fd);
            if (execve(myargv[0], const_cast<char **>(myargv), environ) < 0)
            {
                cout << " execve error" << endl;
                exit(1);
            }
        }
        i++;
    }

    close(fd);

    return 0;
}