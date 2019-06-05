WHICH=$(cat ~/.zshrc | grep "/sgoinfre/goinfre/Perso/malluin/miniconda3/bin" | wc -l)
if [ $WHICH -eq 0 ]
then
	sed -i "" '\|export PATH\=\"\/Users\/malluin\/goinfre\/conda_malluin\/bin:\$PATH\"|d' ~/.zshrc
	echo 'export PATH="/sgoinfre/goinfre/Perso/malluin/miniconda3/bin:$PATH"' >> ~/.zshrc
else
	sed -i "" '\|export PATH\=\"\/sgoinfre\/goinfre\/Perso\/malluin\/miniconda3\/bin:\$PATH\"|d' ~/.zshrc
	echo 'export PATH="/Users/malluin/goinfre/conda_malluin/bin:$PATH"' >> ~/.zshrc
fi
