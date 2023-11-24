#include <bits/stdc++.h>

using namespace std;

class Day{
	private:
    	int u, v;
	public:
    	Day(int uu, int vv) : u(uu), v(vv){}
    	int getU() const{
        	return u;
    	}
    	int getV() const{
        	return v;
    	}
};

class CapNhat{
	private:
    	int u,v,k;
	public:
    	CapNhat(int uu, int vv, int kk) : u(uu), v(vv), k(kk){}
    	int getU() const{
        	return u;
    	}
    	int getV() const{
        	return v;
    	}
    	int getK() const{
       		return k;
    	}
};

class ThemSo{
	private:
    	vector<int> arr;
	public:
    	ThemSo(int n) : arr(n, 0){}
    	void themvao(const CapNhat& capnhat){
        	arr[capnhat.getU() - 1] += capnhat.getK();
        	if (capnhat.getV() < arr.size()) {
            	arr[capnhat.getV()] -= capnhat.getK();
        	}
   		}
    	void calculatePrefixSum(){
        	for (int i = 1; i < arr.size(); ++i){
            	arr[i] += arr[i - 1];
        	}
    	}
    	int getMaxValueInRange(int u, int v) const{
        	int maxVal = INT_MIN;
        	for (int i = u - 1; i < v; ++i){
            	maxVal = max(maxVal, arr[i]);
        	}
        	return maxVal;
    	}	
};

int main(){
    int n, m;
    cin >> n >> m;

    ThemSo add(n);

    for (int i = 0; i < m; ++i){
        int u, v, k;
        cin >> u >> v >> k;
        CapNhat capnhat(u, v, k);
        add.themvao(capnhat);
    }

    add.calculatePrefixSum();

    int p;
    cin >> p;

    for (int i = 0; i < p; ++i){
        int u, v;
        cin >> u >> v;
        Day day(u, v);
        cout << add.getMaxValueInRange(day.getU(), day.getV()) << endl;
    }

    return 0;
}
