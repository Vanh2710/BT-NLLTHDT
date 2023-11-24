#include <iostream>
#include <cstring>
#include <cmath>
using namespace std;
class Tinh{
	private:
		string bieuthuc;
	public:	 
		void Nhap(){
			cin>>bieuthuc;
			cout<<"\n";
		}
		int Test(){
			return 3+(4 - abs(-5))*6;
		}
};
int main() {
   Tinh A;
   A.Nhap();
   cout<<A.Test();
}





