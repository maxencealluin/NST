mkdir ~/goinfre/
cd ~/goinfre/
curl -o miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
printf '\nyes\n~/goinfre/conda/' | sh miniconda3.sh
WHICH=$(cat ~/.zshrc | grep "/sgoinfre/goinfre/Perso/malluin/miniconda3/bin" | wc -l)
if [ $WHICH -eq 0 ]
then
	sed -i "" '\|export PATH\=\"\/Users\/malluin\/goinfre\/conda\/bin:\$PATH\"|d' ~/.zshrc
	echo 'export PATH="/Users/malluin/goinfre/conda/bin:$PATH"' >> ~/.zshrc
else
	sed -i "" '\|export PATH\=\"\/sgoinfre\/goinfre\/Perso\/malluin\/miniconda3\/bin:\$PATH\"|d' ~/.zshrc
	echo 'export PATH="/Users/malluin/goinfre/conda/bin:$PATH"' >> ~/.zshrc
fi
conda env update -f /Users/malluin/IA_projects/ML-env.yml
