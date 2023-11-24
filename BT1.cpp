#include <bits/stdc++.h>

using namespace std;

class Songuyen{
	private:
		int A;
		int B;
	public:
		void Nhap();
		void Cong();
		
};
void Songuyen::Nhap(){
	cout<<"Nhap so nguyen A: "; 
	cin>>A;
	cout<<"Nhap so nguyen B: "; 
	cin>>B;
};
void Songuyen::Cong(){
	cout<< A+B;
};

int main(){
	Songuyen X;
	X.Nhap();
	cout<<"Tong: ";
	X.Cong();
	return 0;
}

