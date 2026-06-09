#!/bin/bash

pthtop="$(cd "$(dirname "${0}")/../../../.." && pwd)"
source "${pthtop}"/manage/lib/params.sh
source "${pthtop}"/manage/lib/shared.sh
source "${pthcrr}"/params.sh

pthapp="${pthsrc}"/envmtv
test -d "${pthapp}" || mkdir "${pthapp}"
cd "${pthapp}" && test -d hgf || mkdir hgf && chmod 777 hgf
cd "${pthapp}" && test -d mes || mkdir mes && chmod 777 mes

addimg ${imgtgt} "${cnfimg}" "${pthdoc}"
