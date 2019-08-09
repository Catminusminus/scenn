RunTests () {
  for file in `\find $1 -maxdepth 1 -type f`; do
    echo $file
    clang++ $file -Wall -Wextra -I$SPROUT_PATH -I$SCENN_PATH -std=gnu++2a -fconstexpr-steps=-1
    ./a.out
  done
}

if [ "1" -gt "$#" ]
then
  echo "an arg needed"
  exit 1
fi
if [ "$#" -gt "1" ]
then
  echo "too many args"
  exit 1
fi
case "$1" in
  "all" ) echo "all" ;;
  * ) RunTests $1 ;;
esac
